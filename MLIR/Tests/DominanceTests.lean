/-
## Test Dominance check
-/

import MLIR.AST
import MLIR.EDSL
import MLIR.Semantics.Dominance
import MLIR.Tests.TestLib
import MLIR.Dialects.ArithSemantics
import MLIR.Dialects.FuncSemantics
import MLIR.Dialects.ScfSemantics


open TestLib
open MLIR.AST

namespace DominanceTests

inductive DominanceTest :=
  | mk {α σ ε} (δ: Dialect α σ ε) (name: String) (expectSuccess: Bool) (region: Region δ): DominanceTest


def DominanceTest.run : DominanceTest -> TestCase
  | DominanceTest.mk _ name expectSuccess region =>
    (name, match singleBBRegionObeySSA region [] with
            | some _ =>
            if expectSuccess then
              .ok ""
            else
              .error "Dominance check succeeded, but expected failure"
            | none =>
            if expectSuccess then
              .error "Dominance check failed, but expected success"
            else
              .ok "")

def trueval :=
    DominanceTest.mk (func_ + arith + scf) "trueval" true [mlir_region| {
  %true = "arith.constant" () {value = 1: i1}: () -> i1
  "scf.assert" (%true) {msg="<FAILED>"}: (i1) -> ()

  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "func.return" (%z): (i32) -> ()
}]

def reuse_val := DominanceTest.mk (func_ + arith + scf) "reuse_val" false [mlir_region| {
  %true = "arith.constant" () {value = 1: i1}: () -> i1
  %true = "arith.constant" () {value = 1: i1}: () -> i1
}]

def use_unknown_op :=
    DominanceTest.mk (func_ + arith + scf) "use_unknown_op" false [mlir_region| {
  "scf.assert" (%true) {msg="<FAILED>"}: (i1) -> ()
}]

def use_after_def :=
    DominanceTest.mk (func_ + arith + scf) "use_after_def" false [mlir_region| {
  "scf.assert" (%true) {msg="<FAILED>"}: (i1) -> ()
  %true = "arith.constant" () {value = 1: i1}: () -> i1
}]

def use_outside_reg :=
    DominanceTest.mk (func_ + arith + scf) "use_outside_reg" false [mlir_region| {
  %b = "arith.constant" () {value = 1: i1}: () -> i1
  "scf.if"(%b) ({}, {
    %c = "arith.constant" () {value = 1: i1}: () -> i1
  }): (i1) -> ()
  "scf.assert" (%c) {msg="<FAILED>"}: (i1) -> ()
}]

def redef_outside_reg :=
    DominanceTest.mk (func_ + arith + scf) "redef_outside_reg" true [mlir_region| {
  %b = "arith.constant" () {value = 1: i1}: () -> i1
  "scf.if"(%b) ({}, {
    %c = "arith.constant" () {value = 1: i1}: () -> i1
  }): (i1) -> ()
  %c = "arith.constant" () {value = 1: i1}: () -> i1
}]

def redef_inside_reg :=
    DominanceTest.mk (func_ + arith + scf) "redef_inside_reg" false [mlir_region| {
  %b = "arith.constant" () {value = 1: i1}: () -> i1
  %c = "arith.constant" () {value = 1: i1}: () -> i1
  "scf.if"(%b) ({}, {
    %c = "arith.constant" () {value = 1: i1}: () -> i1
  }): (i1) -> ()
}]

def redef_inside_op :=
    DominanceTest.mk (func_ + arith + scf) "redef_inside_reg" true [mlir_region| {
  %b = "arith.constant" () {value = 1: i1}: () -> i1
  %y = "scf.if"(%b) ({
    "scf.yield"(%b): (i1) -> ()
  }, {
    %y = "arith.constant" () {value = 1: i1}: () -> i1
    "scf.yield"(%y): (i1) -> ()
  }): (i1) -> (i1)
}]

def redef_next_reg :=
    DominanceTest.mk (func_ + arith + scf) "redef_next_reg" true [mlir_region| {
  %b = "arith.constant" () {value = 1: i1}: () -> i1
  "scf.if"(%b) ({
    %c = "arith.constant" () {value = 1: i1}: () -> i1
  }, {
    %c = "arith.constant" () {value = 1: i1}: () -> i1
  }): (i1) -> ()
}]

def dominanceTests : List DominanceTest :=
  [trueval, reuse_val, use_unknown_op, use_after_def, use_outside_reg,
   redef_outside_reg, redef_inside_reg, redef_inside_op, redef_next_reg]

def testGroup : TestGroup :=
  ("dominance", dominanceTests.map DominanceTest.run)

#eval runTestGroup testGroup

end DominanceTests
