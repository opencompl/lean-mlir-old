import MLIR.EDSL
import MLIR.AST
import MLIR.Dialects.ArithSemantics
import MLIR.Dialects.FuncSemantics
import MLIR.Dialects.ScfSemantics
import MLIR.Dialects.LinalgSemantics
import MLIR.Dialects.CustomTypeSemantics
import MLIR.Tests.TestLib
open MLIR.AST

namespace SemanticsTests

inductive SemanticTest :=
  | mk {α σ ε} (δ: Dialect α σ ε) [S: Semantics δ]:
      String → Region (δ + scf) → SemanticTest

def SemanticTest.name: SemanticTest → String
  | @SemanticTest.mk _ _ _ δ _ str r => str

def SemanticTest.run (t: SemanticTest): Except String String :=
  let (@SemanticTest.mk α σ ε δ S _ r) := t
  let t := denoteRegion (rgn := r) (args := [])
  match t.run (SSAEnv.empty) with
  | .error (msg, state) => .error (s!"error: {msg}\n\tenv: {state}")
  | .ok ((_r, env)) => .ok (s!"ok. final env: {env}")


def trueval := SemanticTest.mk (func_ + arith) "trueval.mlir" [mlir_region| {
  %true = "arith.constant" () {value = 1: i1}: () -> i1
  "scf.assert" (%true) {msg="<FAILED>"}: (i1) -> ()

  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "func.return" (%z): (i32) -> ()
}]

def add := SemanticTest.mk (func_ + arith) "add.mlir" [mlir_region| {
  %r1 = "arith.constant" () {value = 17: i5}: () -> i5
  %r2 = "arith.constant" () {value = 25: i5}: () -> i5
  %r = "arith.addi" (%r1, %r2): (i5, i5) -> i5
  %e = "arith.constant" () {value = 10: i5}: () -> i5
  %b = "arith.cmpi" (%r, %e) {predicate = 0 /- eq -/}: (i5, i5) -> i1
  "scf.assert" (%b) {msg="<FAILED>"}: (i1) -> ()

  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "func.return" (%z): (i32) -> ()
}]

def sub := SemanticTest.mk (func_ + arith) "sub.mlir" [mlir_region| {
  %_1 = "arith.constant" () {value = 8374: i16}: () -> i16
  %_2 = "arith.constant" () {value = 12404: i16}: () -> i16

  %r1 = "arith.subi" (%_2, %_1): (i16, i16) -> i16
  %e1 = "arith.constant" () {value = 4030: i16}: () -> i16
  %b1 = "arith.cmpi" (%r1, %e1) {predicate = 0 /- eq -/}: (i16, i16) -> i1
  "scf.assert" (%b1) {msg="<FAILED>"}: (i1) -> ()

  %r2 = "arith.subi" (%r1, %_1): (i16, i16) -> i16
  %e2 = "arith.constant" () {value = 61192: i16}: () -> i16
  %b2 = "arith.cmpi" (%r1, %e1) {predicate = 0 /- eq -/}: (i16, i16) -> i1
  "scf.assert" (%b2) {msg="<FAILED>"}: (i1) -> ()


  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "func.return" (%z): (i32) -> ()
}]

#print sub

def xor := SemanticTest.mk (func_ + arith) "xor.mlir" [mlir_region| {
  %r1 = "arith.constant" () {value = 17: i8}: () -> i8
  %r2 = "arith.constant" () {value = 25: i8}: () -> i8
  %r = "arith.xori" (%r1, %r2): (i8, i8) -> i8
  %e = "arith.constant" () {value = 8: i8}: () -> i8
  %b = "arith.cmpi" (%r, %e) {predicate = 0 /- eq -/}: (i8, i8) -> i1
  "scf.assert" (%b) {msg="<FAILED>"}: (i1) -> ()

  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "func.return" (%z): (i32) -> ()
}]

def rw1 := SemanticTest.mk (func_ + arith) "rw1.mlir" [mlir_region| {
  %_1 = "arith.constant"() {value = 84: i8}: () -> i8
  %_2 = "arith.constant"() {value = 28: i8}: () -> i8

  %r1 = "arith.xori"(%_1, %_2): (i8, i8) -> i8
  %e1 = "arith.constant"() {value = 72: i8}: () -> i8
  %b1 = "arith.cmpi" (%r1, %e1) {predicate = 0 /- eq -/}: (i8, i8) -> i1
  "scf.assert" (%b1) {msg="<FAILED>"}: (i1) -> ()

  %r2 = "arith.andi"(%_1, %_2): (i8, i8) -> i8
  %e2 = "arith.constant"() {value = 20: i8}: () -> i8
  %b2 = "arith.cmpi" (%r2, %e2) {predicate = 0 /- eq -/}: (i8, i8) -> i1
  "scf.assert" (%b2) {msg="<FAILED>"}: (i1) -> ()

  %r3 = "arith.ori"(%_1, %_2): (i8, i8) -> i8
  %e3 = "arith.constant"() {value = 92: i8}: () -> i8
  %b3 = "arith.cmpi" (%r3, %e3) {predicate = 0 /- eq -/}: (i8, i8) -> i1
  "scf.assert" (%b3) {msg="<FAILED>"}: (i1) -> ()

  %r4 = "arith.addi"(%r1, %r2): (i8, i8) -> i8
  %b4 = "arith.cmpi" (%r4, %r3) {predicate = 0 /- eq -/}: (i8, i8) -> i1
  "scf.assert" (%b4) {msg="<FAILED>"}: (i1) -> ()

  %r5 = "arith.addi"(%_1, %_2): (i8, i8) -> i8
  %e5 = "arith.constant"() {value = 112: i8}: () -> i8
  %b5 = "arith.cmpi" (%r5, %e5) {predicate = 0 /- eq -/}: (i8, i8) -> i1
  "scf.assert" (%b5) {msg="<FAILED>"}: (i1) -> ()

  %r6 = "arith.addi"(%r2, %r3): (i8, i8) -> i8
  %b6 = "arith.cmpi" (%r6, %e5) {predicate = 0 /- eq -/}: (i8, i8) -> i1
  "scf.assert" (%b6) {msg="<FAILED>"}: (i1) -> ()

  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "func.return" (%z): (i32) -> ()
}]

def if_true := SemanticTest.mk (func_ + scf + arith) "if_true.mlir" [mlir_region| {
  %b = "arith.constant" () {value = 1: i1}: () -> i1
  %y = "scf.if"(%b) ({
    %x1 = "arith.constant"() {value = 3: i16}: () -> i16
    "scf.yield"(%x1): (i16) -> ()
  }, {
    %x2 = "arith.constant"() {value = 4: i16}: () -> i16
    "scf.yield"(%x2): (i16) -> ()
  }): (i1) -> i16

  %e = "arith.constant"() {value = 3: i16}: () -> i16
  %b1 = "arith.cmpi" (%y, %e) {predicate = 0 /- eq -/}: (i16, i16) -> i1
  "scf.assert"(%b1) {msg="<FAILED>"}: (i1) -> ()

  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "func.return" (%z): (i32) -> ()
}]

def if_false := SemanticTest.mk (func_ + scf + arith) "if_false.mlir" [mlir_region| {
  %b = "arith.constant" () {value = 0: i1}: () -> i1
  %y = "scf.if"(%b) ({
    %x1 = "arith.constant"() {value = 3: i16}: () -> i16
    "scf.yield"(%x1): (i16) -> ()
  }, {
    %x2 = "arith.constant"() {value = 4: i16}: () -> i16
    "scf.yield"(%x2): (i16) -> ()
  }): (i1) -> i16

  %e = "arith.constant"() {value = 4: i16}: () -> i16
  %b1 = "arith.cmpi" (%y, %e) {predicate = 0 /- eq -/}: (i16, i16) -> i1
  "scf.assert"(%b1) {msg="<FAILED>"}: (i1) -> ()

  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "func.return" (%z): (i32) -> ()
}]

def if_select := SemanticTest.mk (func_ + scf + arith) "if_select.mlir" [mlir_region| {
  %b = "arith.constant" () {value = 1: i1}: () -> i1
  %x1 = "arith.constant"() {value = 12: i16}: () -> i16
  %x2 = "arith.constant"() {value = 16: i16}: () -> i16

  %y1 = "scf.if"(%b) ({
    "scf.yield"(%x1): (i16) -> ()
  }, {
    "scf.yield"(%x2): (i16) -> ()
  }): (i1) -> i16
  %y2 = "arith.select"(%b, %x1, %x2): (i1, i16, i16) -> i16

  %b1 = "arith.cmpi" (%y1, %y2) {predicate = 0 /- eq -/}: (i16, i16) -> i1
  "scf.assert"(%b1) {msg="<y1 == y2>"}: (i1) -> ()

  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "func.return" (%z): (i32) -> ()
}]

-- We only check the return value of the last instruction because we don't
-- yet implement the folding behavior of the loop, and we have no side effects
def for_trivial := SemanticTest.mk (func_ + scf + arith) "for_trivial.mlir" [mlir_region| {
  %lower = "arith.constant"() {value =  8: i32}: () -> i32
  %upper = "arith.constant"() {value = 18: i32}: () -> i32
  %step  = "arith.constant"() {value =  1: i32}: () -> i32

  %r = "scf.for"(%lower, %upper, %step) ({
    "scf.yield"(%step): (i32) -> ()
  }): (i32, i32, i32) -> (i32)

  %e = "arith.constant"() {value = 1: i32}: () -> i32
  %b = "arith.cmpi"(%e, %r) {predicate = 0 /- eq -/}: (i32, i32) -> i1
  "scf.assert"(%b) {msg="<FAILED>"}: (i1) -> ()

  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "func.return" (%z): (i32) -> ()
}]

def for_bound_stepped := SemanticTest.mk (func_ + scf + arith) "for_bound_stepped.mlir" [mlir_region| {
  %lower = "arith.constant"() {value =  0: index}: () -> index
  %upper = "arith.constant"() {value = 18: index}: () -> index
  %step  = "arith.constant"() {value =  3: index}: () -> index

  "scf.for"(%lower, %upper, %step) ({
    ^bb(%i: index):
      %b1 = "arith.cmpi"(%i, %lower) {predicate = 5 /- sge -/}: (index, index) -> i1
      "scf.assert"(%b1) {msg="<assert i >= lower >"}: (i1) -> ()

      %b2 = "arith.cmpi"(%i, %upper) {predicate = 3 /- sle -/}: (index, index) -> i1
      "scf.assert"(%b2) {msg="<assert i <= upper>"}: (i1) -> ()

      "scf.yield"(%step): (index) -> ()
  }): (index, index, index) -> ()

  %z = "arith.constant" () {value = 0: index}: () -> index
  "func.return" (%z): (index) -> ()
}]


def for_bound := SemanticTest.mk (func_ + scf + arith) "for_bound.mlir" [mlir_region| {
  %lower = "arith.constant"() {value =  0: index}: () -> index
  %upper = "arith.constant"() {value = 18: index}: () -> index

  "scf.for'"(%lower, %upper) ({
    ^bb(%i: index):
      %b1 = "arith.cmpi"(%i, %lower) {predicate = 5 /- sge -/}: (index, index) -> i1
      "scf.assert"(%b1) {msg="<[assert i >= lower]>"}: (i1) -> ()
      %b2 = "arith.cmpi"(%i, %upper) {predicate = 3 /- sle -/}: (index, index) -> i1
      "scf.assert"(%b2) {msg="<[assert i <= upper]>"}: (i1) -> ()

      "scf.yield"(%step): (index) -> ()
  }): (index, index, index) -> ()

  %z = "arith.constant" () {value = 0: index}: () -> index
  "func.return" (%z): (index) -> ()
}]


-- TODO: why does this need the manual coercions?
def if_true_custom_type := SemanticTest.mk (CustomTypeDialect + scf + arith) "if_true_custom_type.mlir"
  [mlir_region| {
  %b = "customtype.true" () : () -> ($(.extended (.inl (.inl (.inl CustomType.CBool)))))
  %x1 = "arith.constant"() {value = 12: i16}: () -> i16
  %x2 = "arith.constant"() {value = 16: i16}: () -> i16

  %y = "customtype.if"(%b) ({
    %x11 = "arith.constant"() {value = 3: i16}: () -> i16
    "scf.yield"(%x11): (i16) -> ()
  }, {
    %x22 = "arith.constant"() {value = 4: i16}: () -> i16
    "scf.yield"(%x22): (i16) -> ()
  }): ($(.extended (.inl (.inl (.inl CustomType.CBool))))) -> i16

  %e = "arith.constant"() {value = 3: i16}: () -> i16
  %b1 = "arith.cmpi" (%y, %e) {predicate = 0 /- eq -/}: (i16, i16) -> i1
  "scf.assert"(%b1) {msg="<FAILED>"}: (i1) -> ()

  %z = "arith.constant" () {value = 0: i32}: () -> i32
  "scf.yield" (%z): (i32) -> ()
}]


def semanticTests: List SemanticTest := [
  trueval,
  add,
  sub,
  xor,
  rw1,
  if_true,

  if_false,
  if_select,
  if_true_custom_type,
--  for_trivial,
  for_bound,
  for_bound_stepped
]

open TestLib

def SemanticTest.toTest (t: SemanticTest) : TestCase :=
  match t.run with
  | .ok msg => (t.name, .ok msg)
  | .error err => (t.name, .error err)


def testGroup : TestGroup :=
  ("semantic_test", semanticTests.map (λ t => (t.toTest)))

end SemanticsTests
