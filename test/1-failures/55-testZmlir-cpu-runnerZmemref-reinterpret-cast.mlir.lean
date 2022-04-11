
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
  }) {function_type = (memref<* × f32>) -> (), llvm.emit_c_interface, sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × 3 × f32>
    %3 = "memref.dim"(%2, %0) : (memref<2 × 3 × f32>, index) -> index
    %4 = "memref.dim"(%2, %1) : (memref<2 × 3 × f32>, index) -> index
    "scf.parallel"(%0, %0, %3, %4, %1, %1) ({
    ^bb0(%arg0: index, %arg1: index):
      %6 = "arith.muli"(%arg0, %4) : (index, index) -> index
      %7 = "arith.addi"(%6, %arg1) : (index, index) -> index
      %8 = "arith.index_cast"(%7) : (index) -> i64
      %9 = "arith.sitofp"(%8) : (i64) -> f32
      "memref.store"(%9, %2, %arg0, %arg1) : (f32, memref<2 × 3 × f32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) {operand_segment_sizes = dense<[2, 2, 2, 0]> : vector<4 × i32>} : (index, index, index, index, index, index) -> ()
    %5 = "memref.cast"(%2) : (memref<2 × 3 × f32>) -> memref<* × f32>
    "func.call"(%5) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.call"(%2) {callee = @cast_ranked_memref_to_static_shape} : (memref<2 × 3 × f32>) -> ()
    "func.call"(%2) {callee = @cast_ranked_memref_to_dynamic_shape} : (memref<2 × 3 × f32>) -> ()
    "func.call"(%2) {callee = @cast_unranked_memref_to_static_shape} : (memref<2 × 3 × f32>) -> ()
    "func.call"(%2) {callee = @cast_unranked_memref_to_dynamic_shape} : (memref<2 × 3 × f32>) -> ()
    "memref.dealloc"(%2) : (memref<2 × 3 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<2 × 3 × f32>):
    %0 = "memref.reinterpret_cast"(%arg0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [0], static_sizes = [6, 1], static_strides = [1, 1]} : (memref<2 × 3 × f32>) -> memref<6 × 1 × f32>
    %1 = "memref.cast"(%0) : (memref<6 × 1 × f32>) -> memref<* × f32>
    "func.call"(%1) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × 3 × f32>) -> (), sym_name = "cast_ranked_memref_to_static_shape"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<2 × 3 × f32>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 6 : index} : () -> index
    %3 = "memref.reinterpret_cast"(%arg0, %0, %1, %2, %2, %1) {operand_segment_sizes = dense<[1, 1, 2, 2]> : vector<4 × i32>, static_offsets = [-9223372036854775808], static_sizes = [-1, -1], static_strides = [-9223372036854775808, -9223372036854775808]} : (memref<2 × 3 × f32>, index, index, index, index, index) -> memref<? × ? × f32, #map>
    %4 = "memref.cast"(%3) : (memref<? × ? × f32, #map>) -> memref<* × f32>
    "func.call"(%4) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × 3 × f32>) -> (), sym_name = "cast_ranked_memref_to_dynamic_shape"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<2 × 3 × f32>):
    %0 = "memref.cast"(%arg0) : (memref<2 × 3 × f32>) -> memref<* × f32>
    %1 = "memref.reinterpret_cast"(%0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [0], static_sizes = [6, 1], static_strides = [1, 1]} : (memref<* × f32>) -> memref<6 × 1 × f32>
    %2 = "memref.cast"(%1) : (memref<6 × 1 × f32>) -> memref<* × f32>
    "func.call"(%2) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × 3 × f32>) -> (), sym_name = "cast_unranked_memref_to_static_shape"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<2 × 3 × f32>):
    %0 = "memref.cast"(%arg0) : (memref<2 × 3 × f32>) -> memref<* × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "arith.constant"() {value = 6 : index} : () -> index
    %4 = "memref.reinterpret_cast"(%0, %1, %2, %3, %3, %2) {operand_segment_sizes = dense<[1, 1, 2, 2]> : vector<4 × i32>, static_offsets = [-9223372036854775808], static_sizes = [-1, -1], static_strides = [-9223372036854775808, -9223372036854775808]} : (memref<* × f32>, index, index, index, index, index) -> memref<? × ? × f32, #map>
    %5 = "memref.cast"(%4) : (memref<? × ? × f32, #map>) -> memref<* × f32>
    "func.call"(%5) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × 3 × f32>) -> (), sym_name = "cast_unranked_memref_to_dynamic_shape"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZmlir-cpu-runnerZmemref-reinterpret-cast.out.txt" str
