
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
    %0 = "arith.constant"() {value = 2.00000000 : f32} : () -> f32
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<f32>
    "memref.store"(%0, %1) : (f32, memref<f32>) -> ()
    %2 = "memref.cast"(%1) : (memref<f32>) -> memref<* × f32>
    "func.call"(%2) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "memref.dealloc"(%1) : (memref<f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "print_0d"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 2.00000000 : f32} : () -> f32
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<16 × f32>
    %2 = "memref.cast"(%1) : (memref<16 × f32>) -> memref<? × f32>
    "linalg.fill"(%0, %2) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2 × i32>} : (f32, memref<? × f32>) -> ()
    %3 = "memref.cast"(%2) : (memref<? × f32>) -> memref<* × f32>
    "func.call"(%3) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "memref.dealloc"(%1) : (memref<16 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "print_1d"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 2.00000000 : f32} : () -> f32
    %1 = "arith.constant"() {value = 4.00000000 : f32} : () -> f32
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<3 × 4 × 5 × f32>
    %3 = "memref.cast"(%2) : (memref<3 × 4 × 5 × f32>) -> memref<? × ? × ? × f32>
    "linalg.fill"(%0, %3) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2 × i32>} : (f32, memref<? × ? × ? × f32>) -> ()
    %4 = "arith.constant"() {value = 2 : index} : () -> index
    "memref.store"(%1, %3, %4, %4, %4) : (f32, memref<? × ? × ? × f32>, index, index, index) -> ()
    %5 = "memref.cast"(%3) : (memref<? × ? × ? × f32>) -> memref<* × f32>
    "func.call"(%5) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "memref.dealloc"(%2) : (memref<3 × 4 × 5 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "print_3d"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<* × f32>) -> (), llvm.emit_c_interface, sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1.00000001 : f32} : () -> f32
    %2 = "vector.splat"(%1) : (f32) -> vector<4 × 4 × f32>
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1x1xvector<4 × 4 × f32>>
    "memref.store"(%2, %3, %0, %0) : (vector<4 × 4 × f32>, memref<1x1xvector<4 × 4 × f32>>, index, index) -> ()
    %4 = "memref.cast"(%3) : (memref<1x1xvector<4 × 4 × f32>>) -> memref<?x?xvector<4 × 4 × f32>>
    "func.call"(%4) {callee = @print_memref_vector_4x4xf32} : (memref<?x?xvector<4 × 4 × f32>>) -> ()
    "memref.dealloc"(%3) : (memref<1x1xvector<4 × 4 × f32>>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "vector_splat_2d"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<?x?xvector<4 × 4 × f32>>) -> (), llvm.emit_c_interface, sym_name = "print_memref_vector_4x4xf32", sym_visibility = "private"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZmlir-cpu-runnerZutils.out.txt" str
