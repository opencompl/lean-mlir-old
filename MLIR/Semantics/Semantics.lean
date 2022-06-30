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


inductive BlockResult {α σ ε} (δ: Dialect α σ ε)
| Branch (bb: BBName) (args: List SSAVal)
| Ret (rets:  List ((τ : MLIRType δ) × MLIRType.eval τ))
| Next (val: (τ: MLIRType δ) × τ.eval)

instance : Inhabited (BlockResult δ) where
  default := .Ret []

instance (δ: Dialect α σ ε): ToString (BlockResult δ) where
  toString := fun
    | .Branch bb args => s!"Branch {bb} {args}"
    | .Ret rets       => s!"Ret {rets}"
    | .Next ⟨τ, val⟩  => s!"Next {val}: {τ}"

-- Interpreted operation, like MLIR.AST.Op, but with less syntax
inductive IOp (δ: Dialect α σ ε) := | mk
  (name:    String)
  (args:    List ((τ: MLIRType δ) × τ.eval))
  (bbargs:  List BBName)
  (regions: Nat)
  (attrs:   AttrDict δ)
  (type:    MLIRType δ)

-- Effect to run a region
-- TODO: change this to also deal with scf.if and yield.
inductive RegionE (Δ: Dialect α' σ' ε'): Type -> Type
| RunRegion (ix: Nat): RegionE Δ (BlockResult Δ)

class Semantics (δ: Dialect α σ ε)  where
  -- Events modeling the dialect's computational behavior. Usually operations
  -- are simply denoted by a trigger of such an event. This excludes, however,
  -- operations that have regions or otherwise call into other dialects, since
  -- every operation in the program must be accessible before we start
  -- interpreting.
  E: Type → Type

  -- Operation semantics function: maps an IOp (morally an Op but slightly less
  -- rich to guarantee good properties) to an interaction tree. Usually exposes
  -- any region calls then emits of event of E. This function runs on the
  -- program's entire dialect Δ but returns none for any operation that is not
  -- part of δ.
  semantics_op:
    IOp Δ →
    Option (Fitree (RegionE Δ +' UBE +' E) (BlockResult Δ))

  -- TODO: Allow a dialects' semantics to specify their terminators along with
  -- TODO| their branching behavior, instead of hardcoding it for cf

  -- Event handler used when interpreting the operations and running programs.
  -- This is where most of the computational semantics take place.
  -- TODO: Allow dialect handlers to emit events into other dialects
  handle: E ~> Fitree PVoid


-- The memory of a smaller dialect can be injected into a larger one.

mutual
variable (Δ: Dialect α' σ' ε') [S: Semantics Δ]

def denoteOp (op: Op Δ):
    Fitree (UBE +' SSAEnvE Δ +' S.E) (BlockResult Δ) :=
  match op with
  | .mk name args0 bbargs regions0 attrs (.fn (.tuple τs) t) => do
      -- Read arguments from memory
      let args ← (List.zip args0 τs).mapM (fun (name, τ) => do
          return ⟨τ, ← Fitree.trigger <| SSAEnvE.Get τ name⟩)
      -- Evaluate regions
      let regions := denoteRegions regions0
      -- Built the interpreted operation
      let iop : IOp Δ := IOp.mk name args bbargs regions0.length attrs (.fn (.tuple τs) t)
      -- Use the dialect-provided semantics, and substitute regions
      match S.semantics_op iop with
      | some t =>
          interp (fun _ e =>
            match e with
            | Sum.inl (RegionE.RunRegion ix) => regions.get! ix
            | Sum.inr <| Sum.inl ube => Fitree.trigger ube
            | Sum.inr <| Sum.inr se => Fitree.trigger se
          ) t
      | none => do
          Fitree.trigger <| UBE.DebugUB s!"invalid op: {op}"
          return default

  | _ => do
      Fitree.trigger <| UBE.DebugUB s!"invalid denoteOp: {op}"
      return .Next ⟨.unit, ()⟩

def denoteBBStmt (bbstmt: BasicBlockStmt Δ):
    Fitree (UBE +' SSAEnvE Δ +' S.E) (BlockResult Δ) :=
  match bbstmt with
  | .StmtAssign val _ op => do
      let br ← denoteOp op
      match br with
      | .Next ⟨τ, v⟩ =>
          Fitree.trigger (SSAEnvE.Set τ val v)
      | _ =>
          Fitree.trigger (UBE.DebugUB s!"invalid denoteBBStmt: {bbstmt}")
      return br
  | .StmtOp op =>
      denoteOp op

def denoteBB (bb: BasicBlock Δ):
    Fitree (UBE +' SSAEnvE Δ +' S.E) (BlockResult Δ) :=
  -- TODO: Bind basic block arguments before running the basic block
  -- TODO: Any checks on the BlockResults of intermediate ops?
  match bb with
  | .mk name args [] =>
      return BlockResult.Next ⟨.unit, ()⟩
  | .mk name args [stmt] =>
      denoteBBStmt stmt
  | .mk name args (stmt :: stmts) => do
      let _ ← denoteBBStmt stmt
      denoteBB (.mk name args stmts)

def denoteRegion (r: Region Δ):
    Fitree (UBE +' SSAEnvE Δ +' S.E) (BlockResult Δ) :=
  -- We only define semantics for single-basic-block regions
  -- TODO: Pass region arguments
  -- TODO: Forward region's return type and value
  match r with
  | .mk [bb] =>
      denoteBB bb
  | _ => do
      Fitree.trigger (UBE.DebugUB s!"invalid denoteRegion (>1 bb): {r}")
      return BlockResult.Next ⟨.unit, ()⟩

def denoteRegions (l: List (Region Δ)):
    List (Fitree (UBE +' SSAEnvE Δ +' S.E) (BlockResult Δ)) :=
  match l with
  | [] => []
  | r::l => denoteRegion r :: denoteRegions l
end


instance
    {α₁ σ₁ ε₁} {δ₁: Dialect α₁ σ₁ ε₁}
    {α₂ σ₂ ε₂} {δ₂: Dialect α₂ σ₂ ε₂}
    [S₁: Semantics δ₁]
    [S₂: Semantics δ₂]
    : Semantics (δ₁ + δ₂) where
  E := S₁.E +' S₂.E
  semantics_op op :=
    (S₁.semantics_op op).map (.translate Member.inject) <|>
    (S₂.semantics_op op).map (.translate Member.inject)
  handle := Fitree.case_ S₁.handle S₂.handle

def semanticsRegionRec
    [inst: CoeDialect δ Δ]
    [S: Semantics Δ]
    (fuel: Nat) (r: Region δ) (bb: BasicBlock δ):
    Fitree (UBE +' SSAEnvE Δ +' S.E) (BlockResult Δ) :=
  match fuel with
  | 0 => return .Next ⟨.unit, ()⟩
  | fuel' + 1 => do
      match ← denoteBB Δ bb with
        | .Branch bbname args =>
            -- TODO: Pass the block arguments
            match r.getBasicBlock bbname with
            | some bb' => semanticsRegionRec fuel' r bb'
            | none => return .Next ⟨.unit, ()⟩
        | .Ret rets => return .Ret rets
        | .Next v => return .Next v

-- TODO: Pass region arguments
-- TODO: Forward region's return type and value
def semanticsRegion {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (fuel: Nat) (r: Region Gδ):
    Fitree (UBE +' SSAEnvE Gδ +' S.E) Unit := do
  let _ ← semanticsRegionRec fuel r (r.bbs.get! 0)



def run! {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {R}
    (t: Fitree (UBE +' SSAEnvE Δ +' S.E) R) (env: SSAEnv Δ):
    R × SSAEnv Δ :=
  let t := interp_ub! t
  let t := interp_ssa t env
  let t := interp S.handle t
  t.run

def run {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {R}
    (t: Fitree (UBE +' SSAEnvE Δ +' S.E) R) (env: SSAEnv Δ):
    Option R × SSAEnv Δ :=
  let t := interp_ub t
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

class Denote (δ: Dialect α σ ε) [S: Semantics δ]
    (T: {α σ: Type} → {ε: σ → Type} → Dialect α σ ε → Type) where
  denote: T δ → Fitree (UBE +' SSAEnvE δ +' S.E) (BlockResult δ)

notation "⟦ " t " ⟧" => Denote.denote t

instance (δ: Dialect α σ ε) [Semantics δ]: Denote δ Op where
  denote op := denoteOp δ op

instance (δ: Dialect α σ ε) [Semantics δ]: Denote δ BasicBlockStmt where
  denote bbstmt := denoteBBStmt δ bbstmt

instance (δ: Dialect α σ ε) [Semantics δ]: Denote δ BasicBlock where
  denote bb := denoteBB δ bb


-- Not for regions because we need to specify the fuel
