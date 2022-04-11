
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
  ^bb0(%arg0: memref<7x8x9x10xf32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<5040 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        "affine.for"() ({
        ^bb0(%arg3: index):
          "affine.for"() ({
          ^bb0(%arg4: index):
            "affine.store"(%1, %0, %arg1, %arg2, %arg3, %arg4) {map = #map0} : (f32, memref<5040 × f32>, index, index, index, index) -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        "affine.for"() ({
        ^bb0(%arg3: index):
          "affine.for"() ({
          ^bb0(%arg4: index):
            %2 = "affine.load"(%0, %arg1, %arg2, %arg3, %arg4) {map = #map0} : (memref<5040 × f32>, index, index, index, index) -> f32
            "affine.store"(%2, %arg0, %arg1, %arg2, %arg3, %arg4) {map = #map6} : (f32, memref<7x8x9x10xf32>, index, index, index, index) -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<7x8x9x10xf32>) -> (), sym_name = "unflatten4d"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<8 × 7 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<56 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        "affine.store"(%1, %0, %arg1, %arg2) {map = #map0} : (f32, memref<56 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        %2 = "affine.load"(%0, %arg1, %arg2) {map = #map4} : (memref<56 × f32>, index, index) -> f32
        "affine.store"(%2, %arg0, %arg1, %arg2) {map = #map5} : (f32, memref<8 × 7 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<8 × 7 × f32>) -> (), sym_name = "unflatten2d_with_transpose"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<100 × f32>, %arg1: memref<100 × f32>, %arg2: memref<100 × f32>):
    "affine.for"() ({
    ^bb0(%arg3: index):
      %0 = "affine.load"(%arg1, %arg3) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.store"(%0, %arg0, %arg3) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg3: index):
      %0 = "affine.load"(%arg0, %arg3) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.store"(%0, %arg2, %arg3) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 2 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<100 × f32>, memref<100 × f32>, memref<100 × f32>) -> (), sym_name = "check_src_dst_step"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<64 × 64 × f32, 1>, %arg1: memref<1 × 64 × f32, 1>, %arg2: memref<1 × 64 × f32, 1>):
    %0 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    %1 = "arith.constant"() {value = 1.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        %2 = "affine.for"(%0) ({
        ^bb0(%arg5: index, %arg6: f32):
          %4 = "affine.load"(%arg0, %arg5, %arg4) {map = #map0} : (memref<64 × 64 × f32, 1>, index, index) -> f32
          %5 = "arith.addf"(%arg6, %4) : (f32, f32) -> f32
          "affine.yield"(%5) : (f32) -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (f32) -> f32
        %3 = "arith.addf"(%2, %2) : (f32, f32) -> f32
        "affine.store"(%3, %arg1, %arg3, %arg4) {map = #map0} : (f32, memref<1 × 64 × f32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        %2 = "affine.for"(%1) ({
        ^bb0(%arg5: index, %arg6: f32):
          %4 = "affine.load"(%arg0, %arg5, %arg4) {map = #map0} : (memref<64 × 64 × f32, 1>, index, index) -> f32
          %5 = "arith.mulf"(%arg6, %4) : (f32, f32) -> f32
          "affine.yield"(%5) : (f32) -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : (f32) -> f32
        %3 = "arith.mulf"(%2, %2) : (f32, f32) -> f32
        "affine.store"(%3, %arg2, %arg3, %arg4) {map = #map0} : (f32, memref<1 × 64 × f32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<64 × 64 × f32, 1>, memref<1 × 64 × f32, 1>, memref<1 × 64 × f32, 1>) -> (), sym_name = "reduce_add_non_maximal_f32_f32"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZloop-fusion-4.out.txt" str
