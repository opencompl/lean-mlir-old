/-
## Dialect semantics

This file defines the interface that dialects provide to define their
semantics. This is built upon the `Dialect` interface from `MLIR.Dialects`
which define the custom attributes and type required to model the programs.
-/

import MLIR.Semantics.Fitree
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.AST
open MLIR.AST


inductive BlockResult {Gα Gσ Gε} (Gδ: Dialect Gα Gσ Gε)
| Branch (bb: BBName) (args: List SSAVal)
| Ret (rets: List (SSAVal × MLIRType Gδ))
| Next

class Semantics {α σ ε} (δ: Dialect α σ ε) where
  -- Events modeling the dialect's operations
  E: Type → Type
  -- Monad transformer associated to the dialect's environment
  mT: (Type → Type 1) → (Type → Type 1)
  -- We need to be able to lift
  monad: forall (m: Type → Type 1) [Monad m], Monad (mT m)
  hoist: forall (m: Type → Type 1) (n: Type -> Type 1)
      {α}, ({β: Type} → m β → n β) → mT m α → mT n α
  -- reassoc: forall
  -- (mT: (Type -> Type 1) -> (Type -> Type 1))
  -- (nT: (Type -> Type 1) -> (Type -> Type 1))
  -- (oT: (Type -> Type 1) -> (Type -> Type 1))
  -- (o: Type -> Type 1) {a: Type},
    --  mT (nT o) α → (fun m α => mT (nT )) o α
  -- Semantics.mT δ₁ (Semantics.mT δ₂ (Fitree F)) R : Type 1

  -- (fun m α => Semantics.mT δ₂ (Semantics.mT δ₁ m) α) (Fitree F) R : Type 1

  -- hoist :: Monad m => (forall a. m a -> n a) -> t m b -> t n b
  -- Operation semantics function: maps an `Op` to an interaction tree. Usually
  -- this simply emits an event of `E` and records the return value into the
  -- environment, and could be automated.
  semantics_op {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε}:
    Option SSAVal → Op Gδ → Option (Fitree (SSAEnvE Gδ +' E) (BlockResult Gδ))

  -- TODO: Allow a dialects' semantics to specify their terminators along with
  -- TODO| their branching behavior, instead of hardcoding it for cf

  -- Event handler used when interpreting the operations and running programs.
  -- This is where most of the semantics and computations take place.
  -- TODO: Allow dialect handlers to emit events into other dialects
  interpEvents (F): Fitree (E +' F) ~> mT (Fitree F)

instance {α σ ε} [δ: Dialect α σ ε] [S: Semantics δ] (m) [Monad m]:
    Monad (S.mT _ m) :=
  S.monad m

instance {α₁ σ₁ ε₁} {δ₁: Dialect α₁ σ₁ ε₁} {α₂ σ₂ ε₂} {δ₂: Dialect α₂ σ₂ ε₂}
    [S₁: Semantics δ₁] [S₂: Semantics δ₂]: Semantics (δ₁ + δ₂) where
  E := S₁.E +' S₂.E
  mT m α := S₁.mT _ (S₂.mT _ m) α
  semantics_op ret_name op :=
    (S₁.semantics_op ret_name op).map (.translate Member.inject) <|>
    (S₂.semantics_op ret_name op).map (.translate Member.inject)
  interpEvents F R t :=
    let tr: (S₁.E +' S₂.E) +' F ~> S₁.E +' S₂.E +' F := fun R e =>
      match e with
      | .inl (Sum.inl e) => .inl e
      | .inl (Sum.inr e) => .inr (Sum.inl e)
      | .inr e => .inr (Sum.inr e)
    let t := t.translate tr
    let t := S₁.interpEvents _ _ t
    S₁.hoist _ (Semantics.mT δ₂ _) (fun t' => S₂.interpEvents _ _ t') t

def semantics_op! {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]:
    Option SSAVal → Op Gδ →
    Fitree (UBE +' SSAEnvE Gδ +' S.E) (BlockResult Gδ) :=
  fun ret op =>
    match S.semantics_op ret op with
    | some t => t.translate Member.inject
    | none => do Fitree.trigger (UBE.DebugUB s!"{op}"); return .Next

def semantics_bbstmt {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]:
    BasicBlockStmt Gδ →
    Fitree (UBE +' SSAEnvE Gδ +' S.E) (BlockResult Gδ)
| .StmtAssign val _ op => semantics_op! (some val) op
| .StmtOp op => semantics_op! none op

-- TODO: Add the basic block arguments and bind them before running the block
def semantics_bb {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (bb: BasicBlock Gδ):
    Fitree (UBE +' SSAEnvE Gδ +' S.E) (BlockResult Gδ) := do
  -- TODO: we assume all statements return BlockResult.Next except the last
  for stmt in bb.stmts.init do
    let _ ← semantics_bbstmt stmt
  match bb.stmts.getLast? with
  | some stmt => semantics_bbstmt stmt
  | none => return BlockResult.Next

def semantics_region_go {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (fuel: Nat) (r: Region Gδ) (bb: BasicBlock Gδ):
    Fitree (UBE +' SSAEnvE Gδ +' S.E) (BlockResult Gδ) :=
  match fuel with
  | 0 => return .Next
  | fuel' + 1 => do
      match ← semantics_bb bb with
        | .Branch bbname args =>
            -- TODO: Pass the block arguments
            match r.getBasicBlock bbname with
            | some bb' => semantics_region_go fuel' r bb'
            | none => return .Next
        | .Ret rets => return .Ret rets
        | .Next => return .Next

-- TODO: Pass region arguments
-- TODO: Forward region's return type and value
def semantics_region {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (fuel: Nat) (r: Region Gδ):
    Fitree (UBE +' SSAEnvE Gδ +' S.E) Unit := do
  let _ ← semantics_region_go fuel r (r.bbs.get! 0)

def run {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ] {R}
    (t: Fitree (UBE +' SSAEnvE Gδ +' S.E) R) (env: SSAEnv Gδ):
    R × SSAEnv Gδ :=
  let t := interp_ub! t
  let t := interp_ssa t env
  let t := interp S.handle t
  t.run

def runLogged {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    {R} (t: Fitree (UBE +' SSAEnvE Gδ +' S.E) R) (env: SSAEnv Gδ):
    (R × String) × SSAEnv Gδ :=
  let t := interp_ub! t
  let t := (interp_ssa_logged t).run env
  let t := interp S.handle t
  t.run

/-
### Denotation notation
-/

#print Op
class Denote (δ: Dialect α σ ε) [S: Semantics δ]
    (T: {α σ: Type} → {ε: σ → Type} → Dialect α σ ε → Type) where
  denote: T δ → Fitree (UBE +' SSAEnvE δ +' S.E) (BlockResult δ)

notation "⟦ " t " ⟧" => Denote.denote t

instance (δ: Dialect α σ ε) [Semantics δ]: Denote δ Op where
  denote op := semantics_op! none op

instance (δ: Dialect α σ ε) [Semantics δ]: Denote δ BasicBlockStmt where
  denote := semantics_bbstmt

instance (δ: Dialect α σ ε) [Semantics δ]: Denote δ BasicBlock where
  denote := semantics_bb

-- Not for regions because we need to specify the fuel
