import MLIR.EDSL
import MLIR.AST
import MLIR.Dialects.ArithSemantics
import MLIR.Dialects.ControlFlowSemantics
open MLIR.AST

namespace SemanticsTests

inductive Test :=
  | mk {α σ ε} (δ: Dialect α σ ε) [S: Semantics δ]:
      String → Region (δ + cf) → Test

def Test.name: Test → String
  | @Test.mk _ _ _ δ _ str r => str

def Test.run (t: Test): String :=
  let (@Test.mk α σ ε δ S _ r) := t
  let t := semantics_region 99 r
  let t := interp_ub! t
  let t := interp_ssa t SSAEnv.empty
  let t := interp' S.handle t
  let t := interp ControlFlowOp.handleLogged t
  t.run.run.snd

def trueval := Test.mk arith "trueval.mlir" [mlir_region| {
  %true = "arith.constant" () {value = 1: i1}: () -> i1
  "cf.assert" (%true): (i1) -> ()
}]

def add := Test.mk arith "add.mlir" [mlir_region| {
  %r1 = "arith.constant" () {value = 17: i5}: () -> i5
  %r2 = "arith.constant" () {value = 25: i5}: () -> i5
  %r = "arith.addi" (%r1, %r2): (i5, i5) -> i5
  %e = "arith.constant" () {value = 10: i5}: () -> i5
  %b = "arith.cmpi" (%r, %e) {predicate = 0 /- eq -/}: (i5, i5) -> i1
  "cf.assert" (%b): (i1) -> ()
}]

def allTests: Array Test := #[
  trueval,
  add
]

def runAllTests: IO Bool :=
  allTests.allM (fun t => do
    let b := t.run = ""
    IO.println s!"{t.name}: {if b then "ok" else "FAIL"}"
    return b)

#eval runAllTests

end SemanticsTests
