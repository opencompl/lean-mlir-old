import MLIR.Dialects.ScfSemantics
import MLIR.Dialects.ArithSemantics
open MLIR.AST

namespace SCF_SELECT
abbrev Δ := scf + arith

def LHS: BasicBlockStmt Δ := [mlir_bb_stmt|
  %x = "scf.if" (%b) ({
    "scf.yield"(%n): (i32) -> ()
  }, {
    "scf.yield"(%m): (i32) -> ()
  }): (i1) -> i32
]
def RHS: BasicBlockStmt Δ := [mlir_bb_stmt|
  %x = "arith.select"(%b, %n, %m): (i1, i32, i32) -> i32
]
def INPUT (b: FinInt 1) (n m: FinInt 32): SSAEnv Δ := SSAEnv.One [
  ⟨"b", .i1, b⟩,
  ⟨"n", .i32, n⟩,
  ⟨"m", .i32, m⟩]

theorem denoteYieldRegion:
  denoteRegion Δ (Region.mk
    [BasicBlock.mk bbname []
      [BasicBlockStmt.StmtOp
        (Op.mk "scf.yield" [valuename] [] [] (.mk [])
          (.fn (.tuple [.int .Signless 32]) (.tuple [])))]]) =
  fun (args: TypedArgs Δ) =>
    Fitree.bind (Fitree.trigger <| SSAEnvE.Get MLIRType.i32 valuename) fun x =>
    Fitree.ret (BlockResult.Ret [⟨.i32, x⟩]) := by
  funext args
  simp [denoteRegion, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp]
  simp [List.zip, List.mapM, List.map]
  simp [Semantics.semantics_op]
  simp [HOrElse.hOrElse, OrElse.orElse, Option.orElse, Option.map]
  simp [scf_semantics_op]
  simp [denoteTypedArgs]; cases args <;> simp

theorem equivalent (b: FinInt 1) (n m: FinInt 32):
    (run (denoteBBStmt _ LHS) (INPUT b n m)) =
    (run (denoteBBStmt _ RHS) (INPUT b n m)) := by
  simp [LHS, RHS, INPUT]
  simp [coeMLIRTypeList, coeRegionList, coeBasicBlockStmtList]
  simp [denoteBBStmt, denoteOp, List.zip, List.mapM, denoteRegions]
  simp [Semantics.semantics_op]
  simp [HOrElse.hOrElse, OrElse.orElse, Option.orElse, Option.map]
  simp [scf_semantics_op, arith_semantics_op]
  rw [denoteYieldRegion]
  rw [denoteYieldRegion]
  save

  conv in Fitree.translate _ (Fitree.trigger _) => simp [Fitree.translate]
  conv in Fitree.interp (interpRegion Δ) (Fitree.Vis _ _) =>
    simp [Fitree.interp, interpRegion]
  simp [run, interpSSA'_bind, SSAEnvE.handle, cast_eq]
  simp [Fitree.translate]
  simp [interpRegion, Member.inject, interpSSA'_bind]
  save

  cases b.bool_cases <;> subst b <;> simp [List.get!, interpSSA'_bind]
  . simp [Fitree.trigger, Fitree.interp'_bind, SSAEnvE.handle]
    simp [Fitree.translate, Semantics.handle, cast_eq]
  . simp [Fitree.trigger, Fitree.interp'_bind, SSAEnvE.handle]
    simp [Fitree.translate, Semantics.handle, cast_eq]
end SCF_SELECT
