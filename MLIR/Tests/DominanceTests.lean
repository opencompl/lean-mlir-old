/-
## Test Dominance check
-/

import MLIR.AST
import MLIR.EDSL
import MLIR.Semantics.Dominance
import MLIR.Tests.TestLib
import MLIR.Dialects.ArithSemantics
import MLIR.Dialects.FuncSemantics
import MLIR.Dialects.ControlFlowSemantics


open TestLib
open MLIR.AST

namespace DominanceTests

inductive DominanceTest :=
  | mk {α σ ε} (δ: Dialect α σ ε) (name: String) (expectSuccess: Bool) (region: Region δ): DominanceTest


def DominanceTest.run : DominanceTest -> TestCase
  | DominanceTest.mk _ name expectSuccess region =>
    (name, match singleBBRegionRegionObeySSA region [] with
            | some _ =>
            if expectSuccess then
              .ok ()
            else
              .error "Dominance check succeeded, but expected failure"
            | none =>
            if expectSuccess then
              .error "Dominance check failed, but expected success"
            else
              .ok ())

def trueval := DominanceTest.mk (func_ + arith + cf) "trueval.mlir" true [mlir_region| {
  %true = "arith.constant" () {value = 1: i1}: () -> i1
  --"cf.assert" (%true) {msg="<FAILED>"}: (i1) -> ()

  --%z = "arith.constant" () {value = 0: i32}: () -> i32
  --"func.return" (%z): (i32) -> ()
}]


def dominanceTests : List DominanceTest :=
  [trueval]

def testGroup : TestGroup :=
  ("dominance", dominanceTests.map DominanceTest.run)

#eval runTestGroup testGroup

end DominanceTests