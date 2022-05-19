import MLIR.Dialects.ToyModel
import MLIR.Semantics.Fitree
import MLIR.Semantics.Verifier
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.InvalidOp
import MLIR.Util.Metagen

import MLIR.AST
import MLIR.EDSL

import Lean

open MLIR.AST


set_option hygiene false in
genInductive ControlFlowOp #[
  OpSpec.mk "br" `(
    (target: String) →
    (args: List ((τ: MLIRTy) × τ.eval)) →
    ControlFlowOp Unit)
  , OpSpec.mk "condbr" `(
    (cond: Bool)  →
    ControlFlowOp Unit)
  -- , OpSpec.mk "runRegion" `(
  --    (region: Region) →
  --    (τ: MLIRTy) →
  --    ControlFlowOp (τ.eval))
  ]

#check ControlFlowOp


-- | What's the return type? x(
inductive BranchType
| Br (bbname: BBName)
| Ret


-- | interpret branch and conditional branch
def cf_semantics_op {E: Type -> Type}: Op → Fitree (InvalidOpE +' SSAEnvE +' E) (Option BranchType)
  | Op.mk "br" [] [bbname] [] attrs
        (MLIRTy.fn (MLIRTy.tuple argtys) (MLIRTy.tuple [])) => do
        return (BranchType.Br bbname)
  | Op.mk "condr" [vcond] [bbtrue, bbfalse] [] attrs
        (MLIRTy.fn (MLIRTy.tuple argtys) (MLIRTy.tuple [])) => do
        let condval <- Fitree.trigger $ SSAEnvE.Get (MLIRTy.int 32) vcond
        match condval with
        | 0 => return (BranchType.Br bbtrue)
        | _ => return (BranchType.Br bbfalse)
  | _ => do
      -- TODO: add error messages.
      Fitree.trigger InvalidOpE.InvalidOp
      return Option.none



-- | TODO: generalize so that each dialect can have its effects.
def cf_semantics_bbstmt {E: Type -> Type}:
      BasicBlockStmt → Fitree (InvalidOpE +' SSAEnvE +' E) (Option BranchType)
  | BasicBlockStmt.StmtAssign val _ op => return Option.none
  | BasicBlockStmt.StmtOp op => cf_semantics_op op


-- | TODO: generalize so that each dialect can have its effects.
@[simp]
def cf_semantics_bb {E: Type -> Type} (bb: BasicBlock): Fitree (InvalidOpE +' SSAEnvE +' E) (Option BranchType) :=
  match bb.stmts.getLast? with
  | Option.some stmt => cf_semantics_op (stmt.op)
  | Option.none => return Option.none

-- | The semantics of a region are to use up fuel to run the basic block
-- | as many times as necessary.
def cf_semantics_region_go {E: Type -> Type} (fuel: Nat) (r: Region) (bb: BasicBlock):
  Fitree (InvalidOpE +' SSAEnvE +' E) (Option BranchType) :=
  match fuel with
  | 0 => return Option.none
  | Nat.succ fuel => do
          -- | TODO: refactor using OptionT transformer
          match (<- cf_semantics_bb bb) with
          | Option.none => return Option.none
          | Option.some (BranchType.Ret) => return (BranchType.Ret)
          | Option.some (BranchType.Br bbname) =>
             let bb? := r.getBasicBlock bbname
             match bb? with
             | Option.none => return Option.none
             | Option.some bb => cf_semantics_region_go fuel r bb

-- | TODO: write a procedure to determine the return type of a region!
-- | TODO: we need to run some static analysis to find the type of the region o_O ?
@[simp]
def cf_semantics_region
  {E: Type -> Type} (fuel: Nat) (r: Region) (τ: MLIRTy): Fitree (InvalidOpE +' SSAEnvE +' E) Unit :=
  match fuel with
  | 0 => return ()
  | Nat.succ fuel' => do
     let _ <- cf_semantics_region_go fuel r (r.bbs.get! 0)

/-
TODO: figure out why this does not work

-- | An effect indicating we want to take the fixpoint of some function.
inductive FixpointE: Type _ -> Type _
| Fixpoint {T: Type _} {R: Type _} {M: Type _ -> Type _} (f: T -> M R) (input: T):  FixpointE R

#check Member

def cf_semantics_region_go_fix (r: Region) (bb: BasicBlock):
  Fitree (InvalidOpE +' SSAEnvE +' FixpointE) (Option BranchType) := do
          -- | TODO: inject into monad
          -- let x <- cf_semantics_bb bb
          let x := Option.none
          match x with
          | Option.none => return Option.none
          | Option.some (BranchType.Ret) => return (BranchType.Ret)
          | Option.some (BranchType.Br bbname) =>
             let bb? := r.getBasicBlock bbname
             match bb? with
             | Option.none => return Option.none
             | Option.some bb => do
               let out <- Fitree.trigger (FixpointE.Fixpoint (cf_semantics_region_go_fix r) bb)
               return out
-/
