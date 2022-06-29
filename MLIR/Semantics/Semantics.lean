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
| Ret (rets: List (SSAVal × MLIRType δ))
| Next (val: (τ: MLIRType δ) × τ.eval)

instance [CoeDialect δ Δ] : Coe (BlockResult δ) (BlockResult Δ) where
  coe := sorry

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
-- (regions: List (Fitree (UBE +' SSAEnvE Δ +' ΔE) (BlockResult Δ)))
  (attrs:   AttrDict δ)
  (type:    MLIRType δ)

/-
-- Coercions for IOp into larger dialects from smaller dialects.
-- Required to define semantics injections.
section IOpCoe
def coeTypeValPair
  [Δ: Dialect α σ ε]
  [Δ': Dialect α' σ' ε']
  (xs: List ((τ: MLIRType Δ) × MLIRType.eval τ)):
  List ((τ: MLIRType (Δ + Δ')) × MLIRType.eval τ) := 
   match xs with 
   | .nil => .nil
-- argument v has type 'MLIRType.eval τ : Type'
-- but is expected to have type 'MLIRType.eval (MLIR.AST.coeMLIRType τ) : Type'
--   | .cons ⟨τ, v⟩ xs => .cons ⟨τ, v ⟩ (MLIRValAndTypeInject xs)
   | .cons ⟨τ, v⟩ xs => .cons ⟨τ, sorry⟩ (coeTypeValPair xs)

def IOp.inject_left {Δ': Dialect α' σ' ε'}:
  IOp (Δ: Dialect α σ ε) ΔE -> 
  IOp (Δ + Δ') ΔE
| IOp.mk name args bbargs regions attrs type =>
    IOp.mk name (coeTypeValPair args) bbargs regions attrs type
end IOpCoe
-/
-- Effect to run a region
-- TODO: change this to also deal with scf.if and yield.
inductive RegionE: Type -> Type
| runRegion {T: Type} (ix: Nat): RegionE T

#check Member
-- TODO(sid): do we need external dialects?
-- Semantics of dialect `δ`, which is dependent on external dialects `Δ`.
class Semantics (δ: Dialect α σ ε)  where
  -- Events modeling the dialect's operations.
  E: Type → Type

  -- Operation semantics function: maps an `Op` to an interaction tree. Usually
  -- this simply emits an event of `E` and records the return value into the
  -- environment, and could be automated.
  semantics_op:
    IOp δ →
    Fitree (RegionE +' UBE +' (SSAEnvE δ) +' E) (BlockResult δ)

  -- TODO: Allow a dialects' semantics to specify their terminators along with
  -- TODO| their branching behavior, instead of hardcoding it for cf

  -- Event handler used when interpreting the operations and running programs.
  -- This is where most of the semantics and computations take place.
  -- TODO: Allow dialect handlers to emit events into other dialects
  handle: E ~> Fitree PVoid

-- Given an eliminator for the effect K for *any* T into a
-- *particular R* for (Fitree E R), produce a new (Fitree E R)
def elimEffect (f: Fitree (K +' E) R)
   (eliminator: {T: Type} -> (kt: K T) -> Fitree E R) : Fitree E R :=
  match f with 
  | .Ret r => .Ret r
  | .Vis e k => 
        match e with 
        | .inl cur => eliminator cur
        | .inr rest => .Vis rest (fun t => elimEffect (k t) eliminator)


-- The memory of a smaller dialect can be injected into a larger one.
instance [CoeDialect δ Δ]: Member (SSAEnvE δ) (SSAEnvE Δ) where 
  inject := sorry

mutual
variable (δ: Dialect α σ ε)
variable (Δ: Dialect α' σ' ε')
variable [S: Semantics δ]
variable [CoeDialect δ Δ]

def denoteOp (op: Op δ):
    Fitree (UBE +' SSAEnvE Δ +' S.E) (BlockResult Δ) :=
  match op with
  | .mk name args0 bbargs regions0 attrs (.fn (.tuple τs) t) => do
      -- Read arguments from memory
      let args ← (List.zip args0 τs).mapM (fun (name, τ) => do
          return ⟨τ, ← Fitree.trigger <| SSAEnvE.Get τ name⟩)
      -- Evaluate regions
      let regions := denoteRegions regions0
      -- Built the interpreted operation
      let iop := IOp.mk name args bbargs regions0.length attrs (.fn (.tuple τs) t)
      -- stall iop
      -- Run the dialect-provided semantics
      -- TODO: Coerce the fitree into a larger parent itree
      let childtree 
           : Fitree (RegionE +' UBE +' SSAEnvE δ +' S.E)
                    (BlockResult δ)  := S.semantics_op iop
      -- 1. change result type to (BlockResult Δ) [capital Delta]
      let childtree 
           : Fitree (RegionE +' UBE +' SSAEnvE δ +' S.E)
                    (BlockResult Δ)  := childtree.map Coe.coe
      -- 2. Change environment to  (SSAEnvE Δ)
      -- This will coerce along Fitree.coe_member, since 
      -- (SSAEnvE δ) can be coerced to (SSAEnv Δ)
      let childtree 
           : Fitree (RegionE +' UBE +' SSAEnvE Δ +' S.E)
                    (BlockResult Δ)  := Coe.coe childtree

      let foo := elimEffect childtree 
            (fun re => match re with
                      | RegionE.runRegion ix => regions.get! ix)
      foo

  | _ => do
      Fitree.trigger <| UBE.DebugUB s!"invalid denoteOp: {op}"
      return .Next ⟨.unit, ()⟩

def denoteBBStmt (bbstmt: BasicBlockStmt δ):
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

def denoteBB (bb: BasicBlock δ):
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

def denoteRegion (r: Region δ):
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

def denoteRegions (l: List (Region δ)):
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
  -- semantics_op: IOp (δ+Δ) (E +' ΔE) →
  --  Fitree (RegionE +' UBE +' SSAEnvE (δ+Δ) +' (E +' ΔE)) (BlockResult (δ+Δ))
  -- | sid: how to implement? need to figure out which side to inject
  semantics_op op := sorry /-
    (S₁.semantics_op op).map (.translate Member.inject) <|>
    (S₂.semantics_op op).map (.translate Member.inject)
  -/
  handle := Fitree.case_ S₁.handle S₂.handle

/-
def semantics_region_go {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (fuel: Nat) (r: Region Gδ) (bb: BasicBlock Gδ):
    Fitree (UBE +' SSAEnvE Gδ +' S.E) (BlockResult Gδ) :=
  match fuel with
  | 0 => return .Next ⟨.unit, ()⟩
  | fuel' + 1 => do
      match ← semantics_bb bb with
        | .Branch bbname args =>
            -- TODO: Pass the block arguments
            match r.getBasicBlock bbname with
            | some bb' => semantics_region_go fuel' r bb'
            | none => return .Next ⟨.unit, ()⟩
        | .Ret rets => return .Ret rets
        | .Next v => return .Next v

-- TODO: Pass region arguments
-- TODO: Forward region's return type and value
def semantics_region {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (fuel: Nat) (r: Region Gδ):
    Fitree (UBE +' SSAEnvE Gδ +' S.E) Unit := do
  let _ ← semantics_region_go fuel r (r.bbs.get! 0)
-/


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

/-
class Denote (δ: Dialect α σ ε) [S: Semantics δ]
    (T: {α σ: Type} → {ε: σ → Type} → Dialect α σ ε → Type) where
  denote: T δ → Fitree (UBE +' SSAEnvE δ +' S.E) (BlockResult δ)

notation "⟦ " t " ⟧" => Denote.denote t

instance (δ: Dialect α σ ε) [Semantics δ]: Denote δ Op where
  denote := semantics_op!

instance (δ: Dialect α σ ε) [Semantics δ]: Denote δ BasicBlockStmt where
  denote := semantics_bbstmt

instance (δ: Dialect α σ ε) [Semantics δ]: Denote δ BasicBlock where
  denote := semantics_bb
-/
-- Not for regions because we need to specify the fuel
