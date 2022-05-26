import MLIR.Dialects.ToyModel
import MLIR.Semantics.Fitree
import MLIR.Semantics.Verifier
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.InvalidOp
import MLIR.Util.Metagen
import MLIR.Util.Reduce

import MLIR.AST
import MLIR.EDSL

import Lean

open MLIR.AST


set_option hygiene false in
genInductive ControlFlowOp #[
  OpSpec.mk "cf.br" `(
    (target: String) →
    (args: List ((τ: MLIRTy) × τ.eval)) →
    ControlFlowOp Unit)
  , OpSpec.mk "cf.condbr" `(
    (cond: Bool)  →
    ControlFlowOp Unit)
  -- , OpSpec.mk "runRegion" `(
  --    (region: Region) →
  --    (τ: MLIRTy) →
  --    ControlFlowOp (τ.eval))
  ]



inductive BranchType
| Br (bbname: BBName)
| Ret


-- | interpret branch and conditional branch
def cf_semantics_op {E: Type -> Type}: Op builtin → Fitree (InvalidOpE +' SSAEnvE builtin +' E) (Option BranchType)
  | Op.mk "cf.br" [] [bbname] [] _ _ => do
        return (BranchType.Br bbname)
  | Op.mk "cf.condbr" [vcond] [bbtrue, bbfalse] _ _ _ => do
        let condval <- Fitree.trigger $ SSAEnvE.Get (δ := builtin) (.int 32) vcond
        match condval with
        | 1 => return (BranchType.Br bbtrue)
        | _ => return (BranchType.Br bbfalse)
  | _ => do
      -- TODO: add error messages.
      Fitree.trigger InvalidOpE.InvalidOp
      return Option.none



-- | TODO: generalize so that each dialect can have its effects.
def cf_semantics_bbstmt {E: Type -> Type}:
      BasicBlockStmt builtin → Fitree (InvalidOpE +' SSAEnvE builtin +' E) (Option BranchType)
  | BasicBlockStmt.StmtAssign val _ op => cf_semantics_op op
  | BasicBlockStmt.StmtOp op => cf_semantics_op op


-- | dummy language to check that control flow works correctly.
def dummy_semantics_op {E: Type -> Type} (ret_name: Option SSAVal):
      Op builtin → Fitree (InvalidOpE +' SSAEnvE builtin +' E) Unit
  | Op.mk "dummy.dummy" _ _ _ _ _ =>
        SSAEnv.set? (δ := builtin) (.int 32) ret_name 42
  | Op.mk "dummy.true" _ _ _ _ _ =>
        SSAEnv.set? (δ := builtin) (.int 32) ret_name 1
  | Op.mk "dummy.false" _ _ _ _ _ =>
        SSAEnv.set? (δ := builtin) (.int 32) ret_name 0
  | _ =>
      -- TODO: add error messages.
      Fitree.trigger InvalidOpE.InvalidOp


def dummy_semantics_bbstmt: BasicBlockStmt builtin ->  Fitree (InvalidOpE +' SSAEnvE builtin +' E) Unit
| BasicBlockStmt.StmtAssign val _ op => dummy_semantics_op (some val) op
| BasicBlockStmt.StmtOp op => dummy_semantics_op none op


-- | TODO: generalize so that each dialect can have its effects.
@[simp]
def cf_semantics_bb {E: Type -> Type} (bb: BasicBlock builtin): Fitree (InvalidOpE +' SSAEnvE builtin +' E) (Option BranchType) := do
  for stmt in bb.stmts.init do
     dummy_semantics_bbstmt stmt
  match bb.stmts.getLast? with
  | Option.some stmt => cf_semantics_op (stmt.op)
  | Option.none => return Option.none

-- | The semantics of a region are to use up fuel to run the basic block
-- | as many times as necessary.
def cf_semantics_region_go {E: Type -> Type} (fuel: Nat) (r: Region builtin) (bb: BasicBlock builtin):
  Fitree (InvalidOpE +' SSAEnvE builtin +' E) (Option BranchType) :=
  match fuel with
  | 0 => return Option.none
  | Nat.succ fuel' => do
          -- | TODO: refactor using OptionT transformer
          match (<- cf_semantics_bb bb) with
          | Option.none => return Option.none
          | Option.some (BranchType.Ret) => return (BranchType.Ret)
          | Option.some (BranchType.Br bbname) =>
             let bb? := r.getBasicBlock bbname
             match bb? with
             | Option.none => return Option.none
             | Option.some bb' => cf_semantics_region_go fuel' r bb'

-- | TODO: write a procedure to determine the return type of a region!
-- | TODO: we need to run some static analysis to find the type of the region o_O ?
@[simp]
def cf_semantics_region
  {E: Type -> Type} (fuel: Nat) (r: Region builtin): Fitree (InvalidOpE +' SSAEnvE builtin +' E) Unit := do
     let _ <- cf_semantics_region_go fuel r (r.bbs.get! 0)

@[simp]
def run_cf (t: Fitree (InvalidOpE +' SSAEnvE builtin +' PVoid) Unit) (env: SSAEnv builtin):
    Fitree PVoid ((Unit × String) × SSAEnv builtin) :=
  let x := interp_ssa_logged (interp_invalid t sorry)
  let y := x.run
  let z := y env
  z


/-
### Examples and testing
-/

-- #reduce spins on the dialect coercion because it's mutually inductive, even
-- with skipProofs := true (why?!), so define it as builtin directly
def dummy_stmt: BasicBlockStmt builtin := [mlir_bb_stmt|
  %dummy = "dummy.dummy"() : ()
]

#reduce (skipProofs := true)
  dummy_semantics_bbstmt dummy_stmt


def true_stmt: BasicBlockStmt builtin := [mlir_bb_stmt|
  %true = "dummy.true"() : ()
]

#reduce (skipProofs := true)
  dummy_semantics_bbstmt true_stmt


def false_stmt: BasicBlockStmt builtin := [mlir_bb_stmt|
  %false = "dummy.false"() : ()
]
#reduce (skipProofs := true)
  dummy_semantics_bbstmt false_stmt


def branch_to_true_region: Region builtin := [mlir_region| {
  ^entry:
    %x = "dummy.true"() : ()
    "cf.condbr"(%x) [^bbtrue, ^bbfalse] : ()

  ^bbtrue:
    %y = "dummy.dummy" () : ()
    "ret" () : ()

  ^bbfalse:
    %z = "dummy.dummy" () : ()
    "ret" () : ()

}]

#eval branch_to_true_region

def run_branch_true : String :=
  let x := run_cf (cf_semantics_region 5 branch_to_true_region) SSAEnv.empty
  let y := Fitree.run x
  y.fst.snd

#eval run_branch_true


def branch_to_false_region := [mlir_region| {
  ^entry:
    %x = "dummy.false"() : ()
    "cf.condbr"(%x) [^bbtrue, ^bbfalse] : ()
    -- "cf.br" () [^bbtrue] : ()

  ^bbtrue:
    %y = "dummy.dummy" () : ()
    "ret" () : ()

  ^bbfalse:
    %z = "dummy.dummy" () : ()
    "ret" () : ()

}]

#eval branch_to_false_region

def run_branch_false : String :=
  let x := run_cf (cf_semantics_region 100 branch_to_false_region) SSAEnv.empty
  let y := Fitree.run x
  y.fst.snd

#eval run_branch_false
