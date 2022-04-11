
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
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 256 × f32, #map0>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %1 = "affine.load"(%0, %arg0, %arg1) {map = #map1} : (memref<64 × 256 × f32, #map0>, index, index) -> f32
        "prevent.dce"(%1) : (f32) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "memref.dealloc"(%0) : (memref<64 × 256 × f32, #map0>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "permute"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × f32, #map5>
    %1 = "affine.load"(%0, %arg0) {map = #map6} : (memref<64 × f32, #map5>, index) -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      %2 = "affine.load"(%0, %arg1) {map = #map6} : (memref<64 × f32, #map5>, index) -> f32
      "prevent.dce"(%2) : (f32) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "shift"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 128 × 256 × f32, #map7>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          %1 = "affine.load"(%0, %arg0, %arg1, %arg2) {map = #map8} : (memref<64 × 128 × 256 × f32, #map7>, index, index, index) -> f32
          "prevent.dce"(%1) : (f32) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map9} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "high_dim_permute"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 128 × f32, #map10>
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "invalid_map"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 512 × f32, #map11>
    %1 = "affine.load"(%0, %arg0) {map = #map12} : (memref<64 × 512 × f32, #map11>, index) -> f32
    "prevent.dce"(%1) : (f32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "data_tiling"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 128 × f32, #map13>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %1 = "affine.load"(%0, %arg0, %arg1) {map = #map1} : (memref<64 × 128 × f32, #map13>, index, index) -> f32
        "prevent.dce"(%1) : (f32) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map9} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "strided"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × 5 × f32, #map14>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %1 = "affine.load"(%0, %arg0, %arg1) {map = #map1} : (memref<2 × 5 × f32, #map14>, index, index) -> f32
        "prevent.dce"(%1) : (f32) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map15} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map16} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "strided_cumulative"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "memref.alloc"(%arg0) {operand_segment_sizes = dense<[0, 1]> : vector<2 × i32>} : (index) -> memref<10 × 10 × f32, #map17>
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        %1 = "affine.load"(%0, %arg1, %arg2) {map = #map1} : (memref<10 × 10 × f32, #map17>, index, index) -> f32
        "prevent.dce"(%1) : (f32) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map18} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map18} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "symbolic_operands"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "memref.alloc"(%arg0, %arg1) {operand_segment_sizes = dense<[0, 2]> : vector<2 × i32>} : (index, index) -> memref<256 × 1024 × f32, #map19>
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.for"() ({
      ^bb0(%arg3: index):
        %1 = "affine.load"(%0, %arg2, %arg3) {map = #map1} : (memref<256 × 1024 × f32, #map19>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map20} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index) -> (), sym_name = "semi_affine_layout_map"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {alignment = 32 : i64, operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 128 × 256 × f32, #map7>
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "alignment"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<16 × f64, #map21>, %arg1: f64, %arg2: memref<8 × f64, #map21>, %arg3: memref<24 × f64>):
    %0 = "affine.load"(%arg0) {map = #map2} : (memref<16 × f64, #map21>) -> f64
    %1 = "arith.mulf"(%0, %0) : (f64, f64) -> f64
    "affine.store"(%1, %arg0) {map = #map18} : (f64, memref<16 × f64, #map21>) -> ()
    "func.call"(%arg2) {callee = @single_argument_type} : (memref<8 × f64, #map21>) -> ()
    "func.return"(%arg1) : (f64) -> ()
  }) {function_type = (memref<16 × f64, #map21>, f64, memref<8 × f64, #map21>, memref<24 × f64>) -> f64, sym_name = "multiple_argument_type"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<8 × f64, #map21>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<8 × f64, #map21>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<16 × f64, #map21>
    %2 = "arith.constant"() {value = 2.30000001 : f64} : () -> f64
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<24 × f64>
    "func.call"(%0) {callee = @single_argument_type} : (memref<8 × f64, #map21>) -> ()
    "func.call"(%arg0) {callee = @single_argument_type} : (memref<8 × f64, #map21>) -> ()
    %4 = "func.call"(%1, %2, %0, %3) {callee = @multiple_argument_type} : (memref<16 × f64, #map21>, f64, memref<8 × f64, #map21>, memref<24 × f64>) -> f64
    "func.return"() : () -> ()
  }) {function_type = (memref<8 × f64, #map21>) -> (), sym_name = "single_argument_type"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<8 × f64, #map21>):
    %0 = "arith.constant"() {value = true} : () -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (memref<8 × f64, #map21>) -> i1, sym_name = "non_memref_ret"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<16 × f64, #map21>, %arg1: f64, %arg2: memref<8 × f64, #map21>):
    %0 = "affine.load"(%arg0) {map = #map2} : (memref<16 × f64, #map21>) -> f64
    %1 = "arith.mulf"(%0, %0) : (f64, f64) -> f64
    %2 = "arith.constant"() {value = true} : () -> i1
    "cf.cond_br"(%2)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  
    %3:2 = "func.call"(%arg2) {callee = @ret_single_argument_type} : (memref<8 × f64, #map21>) -> (memref<16 × f64, #map21>, memref<8 × f64, #map21>)
    "func.return"(%3#1, %1) : (memref<8 × f64, #map21>, f64) -> ()
  ^bb2:  
    "func.return"(%arg2, %1) : (memref<8 × f64, #map21>, f64) -> ()
  }) {function_type = (memref<16 × f64, #map21>, f64, memref<8 × f64, #map21>) -> (memref<8 × f64, #map21>, f64), sym_name = "ret_multiple_argument_type"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<8 × f64, #map21>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<8 × f64, #map21>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<16 × f64, #map21>
    %2 = "arith.constant"() {value = 2.30000001 : f64} : () -> f64
    %3:2 = "func.call"(%0) {callee = @ret_single_argument_type} : (memref<8 × f64, #map21>) -> (memref<16 × f64, #map21>, memref<8 × f64, #map21>)
    %4:2 = "func.call"(%arg0) {callee = @ret_single_argument_type} : (memref<8 × f64, #map21>) -> (memref<16 × f64, #map21>, memref<8 × f64, #map21>)
    %5:2 = "func.call"(%1, %2, %0) {callee = @ret_multiple_argument_type} : (memref<16 × f64, #map21>, f64, memref<8 × f64, #map21>) -> (memref<8 × f64, #map21>, f64)
    %6:2 = "func.call"(%5#0) {callee = @ret_single_argument_type} : (memref<8 × f64, #map21>) -> (memref<16 × f64, #map21>, memref<8 × f64, #map21>)
    "func.return"(%1, %0) : (memref<16 × f64, #map21>, memref<8 × f64, #map21>) -> ()
  }) {function_type = (memref<8 × f64, #map21>) -> (memref<16 × f64, #map21>, memref<8 × f64, #map21>), sym_name = "ret_single_argument_type"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<8 × f64, #map21>):
    "func.call"(%arg0) {callee = @func_B} : (memref<8 × f64, #map21>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<8 × f64, #map21>) -> (), sym_name = "func_A"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<8 × f64, #map21>):
    "func.call"(%arg0) {callee = @func_C} : (memref<8 × f64, #map21>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<8 × f64, #map21>) -> (), sym_name = "func_B"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<8 × f64, #map21>):
    "func.return"() : () -> ()
  }) {function_type = (memref<8 × f64, #map21>) -> (), sym_name = "func_C"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<8 × f64, #map21>):
    "func.call"(%arg0) {callee = @some_func_B} : (memref<8 × f64, #map21>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<8 × f64, #map21>) -> (), sym_name = "some_func_A"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<8 × f64, #map21>):
    "test.test"(%arg0) : (memref<8 × f64, #map21>) -> ()
    "func.call"(%arg0) {callee = @some_func_C} : (memref<8 × f64, #map21>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<8 × f64, #map21>) -> (), sym_name = "some_func_B"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<8 × f64, #map21>):
    "func.return"() : () -> ()
  }) {function_type = (memref<8 × f64, #map21>) -> (), sym_name = "some_func_C"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<16 × f64, #map21>) -> (), sym_name = "external_func_A", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<16 × f64, #map21>, f64) -> memref<8 × f64, #map21>, sym_name = "external_func_B", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<16 × f64, #map21>
    "func.call"(%0) {callee = @external_func_A} : (memref<16 × f64, #map21>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "simply_call_external"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<16 × f64, #map21>, %arg1: f64):
    %0 = "func.call"(%arg0, %arg1) {callee = @external_func_B} : (memref<16 × f64, #map21>, f64) -> memref<8 × f64, #map21>
    "func.return"(%0) : (memref<8 × f64, #map21>) -> ()
  }) {function_type = (memref<16 × f64, #map21>, f64) -> memref<8 × f64, #map21>, sym_name = "use_value_of_external"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 2.30000001 : f32} : () -> f32
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<8 × f32, #map21>
    %2 = "affine.parallel"() ({
    ^bb0(%arg0: index):
      "affine.store"(%0, %1, %arg0) {map = #map6} : (f32, memref<8 × f32, #map21>, index) -> ()
      "affine.yield"(%1) : (memref<8 × f32, #map21>) -> ()
    }) {lowerBoundsGroups = dense<1> : tensor<1 × i32>, lowerBoundsMap = #map2, reductions = [2], steps = [1], upperBoundsGroups = dense<1> : tensor<1 × i32>, upperBoundsMap = #map22} : () -> memref<8 × f32, #map21>
    "func.return"(%2) : (memref<8 × f32, #map21>) -> ()
  }) {function_type = () -> memref<8 × f32, #map21>, sym_name = "affine_parallel_norm"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZnormalize-memrefs.out.txt" str
