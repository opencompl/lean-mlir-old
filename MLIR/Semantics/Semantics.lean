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
| Next (val: (τ: MLIRType Gδ) × τ.eval)

instance (Gδ: Dialect Gα Gσ Gε): ToString (BlockResult Gδ) where
  toString := fun
    | .Branch bb args => s!"Branch {bb} {args}"
    | .Ret rets => s!"Ret {rets}"
    | .Next ⟨τ, val⟩ => s!"Next {val}: {τ}"

class Semantics {α σ ε} (δ: Dialect α σ ε) where
  -- Events modeling the dialect's operations
  E: Type → Type

  -- Operation semantics function: maps an `Op` to an interaction tree. Usually
  -- this simply emits an event of `E` and records the return value into the
  -- environment, and could be automated.
  semantics_op {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε}:
    Op Gδ → Option (Fitree (SSAEnvE Gδ +' E) (BlockResult Gδ))

  -- TODO: Allow a dialects' semantics to specify their terminators along with
  -- TODO| their branching behavior, instead of hardcoding it for cf

  -- Event handler used when interpreting the operations and running programs.
  -- This is where most of the semantics and computations take place.
  -- TODO: Allow dialect handlers to emit events into other dialects
  handle: E ~> Fitree PVoid

instance {α₁ σ₁ ε₁} {δ₁: Dialect α₁ σ₁ ε₁} {α₂ σ₂ ε₂} {δ₂: Dialect α₂ σ₂ ε₂}
    [S₁: Semantics δ₁] [S₂: Semantics δ₂]: Semantics (δ₁ + δ₂) where
  E := S₁.E +' S₂.E
  semantics_op op :=
    (S₁.semantics_op op).map (.translate Member.inject) <|>
    (S₂.semantics_op op).map (.translate Member.inject)
  handle := Fitree.case_ S₁.handle S₂.handle

def semantics_op! {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (op: Op Gδ): Fitree (UBE +' SSAEnvE Gδ +' S.E) (BlockResult Gδ) :=
  match S.semantics_op op with
  | some t => t.translate Member.inject
  | none => do
    Fitree.trigger (UBE.DebugUB s!"{op}")
    return .Ret []

def semantics_bbstmt {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (bbstmt: BasicBlockStmt Gδ):
    Fitree (UBE +' SSAEnvE Gδ +' S.E) (BlockResult Gδ) :=
  match bbstmt with
  | .StmtAssign val _ op => do
      let br ← semantics_op! op
      match br with
      | .Next ⟨τ, v⟩ =>
          Fitree.trigger (SSAEnvE.Set (δ := Gδ) τ val v)
      | _ =>
          Fitree.trigger (UBE.DebugUB s!"invalid assignment: {bbstmt}")
      return br
  | .StmtOp op =>
      semantics_op! op

-- TODO: Add the basic block arguments and bind them before running the block
def semantics_bb {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (bb: BasicBlock Gδ):
    Fitree (UBE +' SSAEnvE Gδ +' S.E) (BlockResult Gδ) := do
  -- TODO: we assume all statements return BlockResult.Next except the last
  for stmt in bb.stmts.init do
    let _ ← semantics_bbstmt stmt
  match bb.stmts.getLast? with
  | some stmt => semantics_bbstmt stmt
  | none => return BlockResult.Next ⟨.unit, ()⟩

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

-- TODO: Pass region arguments
-- TODO: Forward region's return type and value
-- Run the first basic block of the region.
-- This is faithful if we are guaranteed that the region has only
-- a single BB.
def semantics_region_single_bb {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (r: Region Gδ):
    Fitree (UBE +' SSAEnvE Gδ +' S.E) Unit := do
  let _ ← semantics_bb (r.bbs.get! 0)

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

-- Not for regions because we need to specify the fuel
