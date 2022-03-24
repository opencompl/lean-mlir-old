
import MLIR.Doc
import MLIR.AST
import MLIR.EDSL

open  MLIR.EDSL
open MLIR.Doc
open IO

-- | write an op into the path
def o: Op := [mlir_op|
    "builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2xf32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>, test.ptr = "alloc"} : () -> memref<8x64xf32>
    "memref.dealloc"(%0) {test.ptr = "dealloc"} : (memref<8x64xf32>) -> ()
    "func.return"() {test.ptr = "return"} : () -> ()
  }) {function_type = (memref<2xf32>) -> (), sym_name = "no_side_effects", test.ptr = "func"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<i32>, %arg1: i32):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2xi32>, test.ptr = "alloc"} : () -> memref<i32>
    "memref.store"(%arg1, %0) {test.ptr = "store"} : (i32, memref<i32>) -> ()
    %1 = "memref.load"(%0) {test.ptr = "load"} : (memref<i32>) -> i32
    "func.return"() {test.ptr = "return"} : () -> ()
  }) {function_type = (memref<i32>, i32) -> (), sym_name = "simple", test.ptr = "func"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<i32>, %arg1: memref<i32>, %arg2: i32):
    "memref.store"(%arg2, %arg1) {test.ptr = "store"} : (i32, memref<i32>) -> ()
    %0 = "memref.load"(%arg1) {test.ptr = "load"} : (memref<i32>) -> i32
    "func.return"() {test.ptr = "return"} : () -> ()
  }) {function_type = (memref<i32>, memref<i32>, i32) -> (), sym_name = "mayalias", test.ptr = "func"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<i32>, %arg1: memref<i32>, %arg2: i1, %arg3: i32):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2xi32>, test.ptr = "alloc"} : () -> memref<i32>
    "scf.if"(%arg2) ({
      "memref.store"(%arg3, %arg0) : (i32, memref<i32>) -> ()
      %1 = "memref.load"(%arg0) : (memref<i32>) -> i32
      "scf.yield"() : () -> ()
    }, {
    }) {test.ptr = "if"} : (i1) -> ()
    "func.return"() {test.ptr = "return"} : () -> ()
  }) {function_type = (memref<i32>, memref<i32>, i1, i32) -> (), sym_name = "recursive", test.ptr = "func"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<i32>):
    "foo.op"() {test.ptr = "unknown"} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<i32>) -> (), sym_name = "unknown", test.ptr = "func"} : () -> ()
}) : () -> ()


] 
-- | main program
def main : IO Unit :=
    let str := Pretty.doc o
    FS.writeFile "test-alias-analysis-modref.out.txt" str
