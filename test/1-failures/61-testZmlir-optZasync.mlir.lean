
import MLIR.Doc
import MLIR.AST
import MLIR.EDSL

open Lean
open Lean.Parser
open  MLIR.EDSL
open MLIR.AST
open MLIR.Doc
open IO

set_option maxHeartbeats 999999999


declare_syntax_cat mlir_ops
syntax (ws mlir_op ws)* : mlir_ops
syntax "[mlir_ops|" mlir_ops "]" : term

macro_rules
|`([mlir_ops| $[ $xs ]* ]) => do 
  let xs <- xs.mapM (fun x =>`([mlir_op| $x]))
  quoteMList xs.toList (<-`(MLIR.AST.Op))

  
-- | write an op into the path
def o: List Op := [mlir_ops|
"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 2 : index} : () -> index
    %3 = "arith.constant"() {value = 3 : index} : () -> index
    %4 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    %5 = "arith.constant"() {value = 1.00000000 : f32} : () -> f32
    %6 = "arith.constant"() {value = 2.00000000 : f32} : () -> f32
    %7 = "arith.constant"() {value = 3.00000000 : f32} : () -> f32
    %8 = "arith.constant"() {value = 4.00000000 : f32} : () -> f32
    %9 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<4 × f32>
    "linalg.fill"(%4, %9) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2 × i32>} : (f32, memref<4 × f32>) -> ()
    %10 = "memref.cast"(%9) : (memref<4 × f32>) -> memref<* × f32>
    "func.call"(%10) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "memref.store"(%5, %9, %0) : (f32, memref<4 × f32>, index) -> ()
    "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
    "func.call"(%10) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    %11 = "async.execute"() ({
      "memref.store"(%6, %9, %1) : (f32, memref<4 × f32>, index) -> ()
      "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
      "func.call"(%10) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
      %12 = "async.execute"() ({
        "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
        "async.yield"() : () -> ()
      }) {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> !async.token
      %13 = "async.execute"(%12) ({
        "memref.store"(%7, %9, %2) : (f32, memref<4 × f32>, index) -> ()
        "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
        "func.call"(%10) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
        "async.yield"() : () -> ()
      }) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (!async.token) -> !async.token
      "async.await"(%13) : (!async.token) -> ()
      "memref.store"(%8, %9, %3) : (f32, memref<4 × f32>, index) -> ()
      "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
      "func.call"(%10) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
      "async.yield"() : () -> ()
    }) {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> !async.token
    "async.await"(%11) : (!async.token) -> ()
    "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
    "func.call"(%10) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "memref.dealloc"(%9) : (memref<4 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "mlirAsyncRuntimePrintCurrentThreadId", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<* × f32>) -> (), llvm.emit_c_interface, sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZmlir-optZasync.out.txt" str
