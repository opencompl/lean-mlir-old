
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
  "gpu.module"() ({
    "gpu.func"() ({
    ^bb0(%arg0: memref<3 × f32>, %arg1: memref<3 × 3 × f32>, %arg2: memref<3 × 3 × 3 × f32>):
      %0 = "arith.constant"() {value = 0 : index} : () -> index
      %1 = "arith.constant"() {value = 1 : index} : () -> index
      %2 = "arith.constant"() {value = 2 : index} : () -> index
      %3 = "memref.load"(%arg0, %0) : (memref<3 × f32>, index) -> f32
      %4 = "memref.load"(%arg1, %0, %0) : (memref<3 × 3 × f32>, index, index) -> f32
      %5 = "arith.addf"(%3, %4) : (f32, f32) -> f32
      "memref.store"(%5, %arg2, %0, %0, %0) : (f32, memref<3 × 3 × 3 × f32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %0, %1, %0) : (f32, memref<3 × 3 × 3 × f32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %0, %2, %0) : (f32, memref<3 × 3 × 3 × f32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %1, %0, %1) : (f32, memref<3 × 3 × 3 × f32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %1, %1, %1) : (f32, memref<3 × 3 × 3 × f32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %1, %2, %1) : (f32, memref<3 × 3 × 3 × f32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %2, %0, %2) : (f32, memref<3 × 3 × 3 × f32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %2, %1, %2) : (f32, memref<3 × 3 × 3 × f32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %2, %2, %2) : (f32, memref<3 × 3 × 3 × f32>, index, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {function_type = (memref<3 × f32>, memref<3 × 3 × f32>, memref<3 × 3 × 3 × f32>) -> (), gpu.kernel, spv.entry_point_abi = {local_size = dense<1> : vector<3 × i32>}, sym_name = "sum", workgroup_attributions = 0 : i64} : () -> ()
    "gpu.module_end"() : () -> ()
  }) {sym_name = "kernels"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<3 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<3 × 3 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<3 × 3 × 3 × f32>
    %3 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    %4 = "arith.constant"() {value = 3.40000000 : f32} : () -> f32
    %5 = "arith.constant"() {value = 4.30000000 : f32} : () -> f32
    %6 = "memref.cast"(%0) : (memref<3 × f32>) -> memref<? × f32>
    %7 = "memref.cast"(%1) : (memref<3 × 3 × f32>) -> memref<? × ? × f32>
    %8 = "memref.cast"(%2) : (memref<3 × 3 × 3 × f32>) -> memref<? × ? × ? × f32>
    "func.call"(%6, %4) {callee = @fillF32Buffer1D} : (memref<? × f32>, f32) -> ()
    "func.call"(%7, %5) {callee = @fillF32Buffer2D} : (memref<? × ? × f32>, f32) -> ()
    "func.call"(%8, %3) {callee = @fillF32Buffer3D} : (memref<? × ? × ? × f32>, f32) -> ()
    %9 = "arith.constant"() {value = 1 : index} : () -> index
    "gpu.launch_func"(%9, %9, %9, %9, %9, %9, %0, %1, %2) {kernel = @kernels::@sum, operand_segment_sizes = dense<[0, 1, 1, 1, 1, 1, 1, 0, 3]> : vector<9 × i32>} : (index, index, index, index, index, index, memref<3 × f32>, memref<3 × 3 × f32>, memref<3 × 3 × 3 × f32>) -> ()
    %10 = "memref.cast"(%2) : (memref<3 × 3 × 3 × f32>) -> memref<* × f32>
    "func.call"(%10) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<? × f32>, f32) -> (), sym_name = "fillF32Buffer1D", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<? × ? × f32>, f32) -> (), sym_name = "fillF32Buffer2D", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<? × ? × ? × f32>, f32) -> (), sym_name = "fillF32Buffer3D", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<* × f32>) -> (), sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
}) {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_8bit_storage]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3 × i32>}>} : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZmlir-spirv-cpu-runnerZsimple_add.out.txt" str
