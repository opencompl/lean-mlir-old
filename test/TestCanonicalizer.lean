
import MLIR.Doc
import MLIR.AST
import MLIR.EDSL

open Lean
open Lean.Parser
open  MLIR.EDSL
open MLIR.AST
open MLIR.Doc
open IO

declare_syntax_cat mlir_ops
syntax (ws mlir_op ws)* : mlir_ops
syntax "[mlir_ops|" mlir_ops "]" : term

macro_rules
| `([mlir_ops| $[ $xs ]* ]) => do 
  let xs <- xs.mapM (fun x => `([mlir_op| $x]))
  quoteMList xs.toList

-- | write an op into the path
def o: List Op := [mlir_ops|
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc"} : () -> memref<8 × 64 × f32>
    "memref.dealloc"(%0) {test.ptr = "dealloc"} : (memref<8 × 64 × f32>) -> ()
    "func.return"() {test.ptr = "return"} : () -> ()
  }) {function_type = (memref<2 × f32>) -> (), sym_name = "no_side_effects", test.ptr = "func"} : () -> ()
}) : () -> ()

"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<i32>, %arg1: i32):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc"} : () -> memref<i32>
    "memref.store"(%arg1, %0) {test.ptr = "store"} : (i32, memref<i32>) -> ()
    %1 = "memref.load"(%0) {test.ptr = "load"} : (memref<i32>) -> i32
    "func.return"() {test.ptr = "return"} : () -> ()
  }) {function_type = (memref<i32>, i32) -> (), sym_name = "simple", test.ptr = "func"} : () -> ()
}) : () -> ()

"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<i32>, %arg1: memref<i32>, %arg2: i32):
    "memref.store"(%arg2, %arg1) {test.ptr = "store"} : (i32, memref<i32>) -> ()
    %0 = "memref.load"(%arg1) {test.ptr = "load"} : (memref<i32>) -> i32
    "func.return"() {test.ptr = "return"} : () -> ()
  }) {function_type = (memref<i32>, memref<i32>, i32) -> (), sym_name = "mayalias", test.ptr = "func"} : () -> ()
}) : () -> ()

"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<i32>, %arg1: memref<i32>, %arg2: i1, %arg3: i32):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc"} : () -> memref<i32>
    "scf.if"(%arg2) ({
      "memref.store"(%arg3, %arg0) : (i32, memref<i32>) -> ()
      %1 = "memref.load"(%arg0) : (memref<i32>) -> i32
      "scf.yield"() : () -> ()
    }, {
    }) {test.ptr = "if"} : (i1) -> ()
    "func.return"() {test.ptr = "return"} : () -> ()
  }) {function_type = (memref<i32>, memref<i32>, i1, i32) -> (), sym_name = "recursive", test.ptr = "func"} : () -> ()
}) : () -> ()

"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<i32>):
    "foo.op"() {test.ptr = "unknown"} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<i32>) -> (), sym_name = "unknown", test.ptr = "func"} : () -> ()
}) : () -> ()


] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZAnalysisZtest-alias-analysis-modref.out.txt" str
