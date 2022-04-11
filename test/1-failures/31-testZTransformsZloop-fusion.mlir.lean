
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
^bb0:
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_raw_dep_for_locality"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
        %5 = "affine.load"(%0, %arg0, %arg1) {map = #map1} : (memref<10 × 10 × f32>, index, index) -> f32
        %6 = "arith.addf"(%4, %5) : (f32, f32) -> f32
        "affine.store"(%6, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%4, %2, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_reduction_to_pointwise"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.store"(%1, %0, %arg0, %arg1) {map = #map0} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.load"(%0, %arg0, %arg1) {map = #map3} : (memref<10 × 10 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_loop_nests_with_shifts"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.store"(%2, %0, %arg0, %arg1) {map = #map0} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %3 = "affine.load"(%0, %arg1, %arg0) {map = #map0} : (memref<10 × 10 × f32>, index, index) -> f32
        "affine.store"(%3, %1, %arg0, %arg1) {map = #map0} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %3 = "affine.load"(%1, %arg0, %arg1) {map = #map0} : (memref<10 × 10 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_loop_nest"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%4, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%3, %2, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_across_intermediate_loop_with_no_deps"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_all_loops"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%3, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%2, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_first_and_second_loops"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%3, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      %4 = "affine.load"(%2, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %2, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_not_fuse_would_create_cycle"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_producer_consumer"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_and_move_to_preserve_war_dep"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    %2 = "arith.constant"() {value = 4 : index} : () -> index
    %3 = "affine.load"(%0, %2) {map = #map0} : (memref<10 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_if_top_level_access"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    %2 = "affine.load"(%0) {map = #map4} : (memref<100 × f32>) -> f32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_but_not_remove_src"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_no_top_level_access"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    %2 = "arith.constant"() {value = 4 : index} : () -> index
    "affine.if"(%2) ({
      "affine.yield"() : () -> ()
    }, {
    }) {condition = #set} : (index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_not_fuse_if_op_at_top_level"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %2 = "arith.constant"() {value = 4 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.if"(%2) ({
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set} : (index) -> ()
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_not_fuse_if_op_in_loop_nest"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.if"(%arg0) ({
        %3 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"(%0) : (memref<10 × f32>) -> ()
  }) {function_type = () -> memref<10 × f32>, sym_name = "should_fuse_if_op_in_loop_nest_not_sandwiched"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.if"(%arg0) ({
        "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"(%1) : (memref<10 × f32>) -> ()
  }) {function_type = () -> memref<10 × f32>, sym_name = "should_not_fuse_if_op_in_loop_nest_between_src_and_dest"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 20 × 30 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          "affine.store"(%1, %0, %arg0, %arg1, %arg2) {map = #map0} : (f32, memref<10 × 20 × 30 × f32>, index, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          %2 = "affine.load"(%0, %arg1, %arg2, %arg0) {map = #map0} : (memref<10 × 20 × 30 × f32>, index, index, index) -> f32
          "foo"(%2) : (f32) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "permute_and_fuse"} : () -> ()
}) : () -> ()









"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<64 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<16 × 4 × f32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      %1 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<64 × f32>, index) -> f32
      "affine.store"(%1, %0, %arg1) {map = #map1} : (f32, memref<16 × 4 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        %1 = "affine.load"(%0, %arg1, %arg2) {map = #map4} : (memref<16 × 4 × f32>, index, index) -> f32
        "foo"(%1) : (f32) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<64 × f32>) -> (), sym_name = "fuse_reshape_64_16_4"} : () -> ()
}) : () -> ()









"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<16 × 4 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × f32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<16 × 4 × f32>, index, index) -> f32
        "affine.store"(%2, %1, %arg0, %arg1) {map = #map1} : (f32, memref<64 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.load"(%1, %arg0) {map = #map5} : (memref<64 × f32>, index) -> f32
      "foo"(%2) : (f32) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "fuse_reshape_16_4_64"} : () -> ()
}) : () -> ()


















"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2x2x3x3x16x1xi32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 9 × i32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 9 × i32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          "affine.for"() ({
          ^bb0(%arg3: index):
            "affine.for"() ({
            ^bb0(%arg4: index):
              "affine.for"() ({
              ^bb0(%arg5: index):
                %3 = "foo"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (index, index, index, index, index, index) -> i32
                "affine.store"(%3, %0, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {map = #map0} : (i32, memref<2x2x3x3x16x1xi32>, index, index, index, index, index, index) -> ()
                "affine.yield"() : () -> ()
              }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
              "affine.yield"() : () -> ()
            }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %3 = "affine.apply"(%arg0, %arg1) {map = #map6} : (index, index) -> index
        %4 = "affine.apply"(%3) {map = #map7} : (index) -> index
        %5 = "affine.apply"(%3) {map = #map8} : (index) -> index
        %6 = "affine.apply"(%3) {map = #map9} : (index) -> index
        %7 = "affine.apply"(%3) {map = #map10} : (index) -> index
        %8 = "affine.apply"(%3) {map = #map11} : (index) -> index
        %9 = "affine.apply"(%3) {map = #map12} : (index) -> index
        %10 = "affine.load"(%0, %4, %5, %6, %7, %8, %9) {map = #map0} : (memref<2x2x3x3x16x1xi32>, index, index, index, index, index, index) -> i32
        "affine.store"(%10, %1, %arg0, %arg1) {map = #map13} : (i32, memref<64 × 9 × i32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map14} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map15} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %3 = "affine.load"(%1, %arg0, %arg1) {map = #map13} : (memref<64 × 9 × i32>, index, index) -> i32
        %4 = "arith.muli"(%3, %3) : (i32, i32) -> i32
        "affine.store"(%4, %2, %arg0, %arg1) {map = #map13} : (i32, memref<64 × 9 × i32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map14} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map15} : () -> ()
    "func.return"(%2) : (memref<64 × 9 × i32>) -> ()
  }) {function_type = () -> memref<64 × 9 × i32>, sym_name = "R6_to_R2_reshape_square"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "affine.apply"(%arg1) {map = #map0} : (index) -> index
    %1 = "memref.alloc"(%arg0, %0) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    %2 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    %3 = "arith.constant"() {value = 5 : index} : () -> index
    "affine.for"(%arg0) ({
    ^bb0(%arg2: index):
      "affine.for"(%arg1) ({
      ^bb0(%arg3: index):
        "affine.store"(%2, %1, %arg2, %arg3) {map = #map1} : (f32, memref<? × ? × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map0} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : (index) -> ()
    "affine.for"(%arg0) ({
    ^bb0(%arg2: index):
      "affine.for"(%arg1) ({
      ^bb0(%arg3: index):
        %4 = "affine.load"(%1, %arg2, %arg3, %3) {map = #map4} : (memref<? × ? × f32>, index, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : (index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index) -> (), sym_name = "fuse_symbolic_bounds"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 100 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
        %3 = "affine.load"(%0, %arg0, %arg1) {map = #map1} : (memref<10 × 100 × f32>, index, index) -> f32
        %4 = "maxf"(%2, %3) : (f32, f32) -> f32
        "affine.store"(%4, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
        %3 = "affine.load"(%0, %arg0, %arg1) {map = #map1} : (memref<10 × 100 × f32>, index, index) -> f32
        %4 = "arith.subf"(%3, %2) : (f32, f32) -> f32
        "affine.store"(%4, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_reduction_at_depth_of_one"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × 16 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × 16 × f32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<100 × 16 × f32>, index, index) -> f32
        "op0"(%2) : (f32) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "op1"() : () -> f32
        "affine.store"(%2, %1, %arg0, %arg1) {map = #map0} : (f32, memref<100 × 16 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.load"(%1, %arg0, %arg1) {map = #map0} : (memref<100 × 16 × f32>, index, index) -> f32
        "op2"(%2) : (f32) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_at_src_depth1_and_dst_depth1"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.apply"(%arg0, %arg1) {map = #map3} : (index, index) -> index
        %3 = "affine.load"(%0, %2) {map = #map0} : (memref<100 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_src_depth1_at_dst_depth2"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %1) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "fusion_at_depth0_not_currently_supported"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2x2x3x3x16x10xf32, 2>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2x2x3x3x16x10xf32, 2>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<3x3x3x3x16x10xf32, 2>
    %3 = "arith.constant"() {value = 0 : index} : () -> index
    %4 = "arith.constant"() {value = 1 : index} : () -> index
    %5 = "arith.constant"() {value = 1 : index} : () -> index
    %6 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          "affine.for"() ({
          ^bb0(%arg3: index):
            "affine.for"() ({
            ^bb0(%arg4: index):
              "affine.for"() ({
              ^bb0(%arg5: index):
                %7 = "affine.load"(%0, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {map = #map0} : (memref<2x2x3x3x16x10xf32, 2>, index, index, index, index, index, index) -> f32
                "affine.yield"() : () -> ()
              }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
              "affine.yield"() : () -> ()
            }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
            "affine.for"() ({
            ^bb0(%arg4: index):
              "affine.for"() ({
              ^bb0(%arg5: index):
                "affine.store"(%6, %1, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {map = #map0} : (f32, memref<2x2x3x3x16x10xf32, 2>, index, index, index, index, index, index) -> ()
                "affine.yield"() : () -> ()
              }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
              "affine.yield"() : () -> ()
            }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          "affine.for"() ({
          ^bb0(%arg3: index):
            "affine.for"() ({
            ^bb0(%arg4: index):
              "affine.for"() ({
              ^bb0(%arg5: index):
                "affine.for"() ({
                ^bb0(%arg6: index):
                  "affine.for"() ({
                  ^bb0(%arg7: index):
                    "affine.for"() ({
                    ^bb0(%arg8: index):
                      "affine.for"() ({
                      ^bb0(%arg9: index):
                        %7 = "affine.load"(%0, %arg6, %arg7, %arg4, %arg5, %arg8, %arg9) {map = #map0} : (memref<2x2x3x3x16x10xf32, 2>, index, index, index, index, index, index) -> f32
                        "affine.yield"() : () -> ()
                      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
                      "affine.yield"() : () -> ()
                    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
                    "affine.for"() ({
                    ^bb0(%arg8: index):
                      "affine.for"() ({
                      ^bb0(%arg9: index):
                        %7 = "affine.load"(%1, %arg2, %arg3, %arg0, %arg1, %arg8, %arg9) {map = #map0} : (memref<2x2x3x3x16x10xf32, 2>, index, index, index, index, index, index) -> f32
                        "affine.yield"() : () -> ()
                      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
                      "affine.yield"() : () -> ()
                    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
                    "affine.yield"() : () -> ()
                  }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
                  "affine.yield"() : () -> ()
                }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
                "affine.yield"() : () -> ()
              }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
              "affine.yield"() : () -> ()
            }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_deep_loop_nests"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<4 × 256 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<4 × 256 × f32>
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    %3 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %4 = "affine.load"(%1, %arg0, %arg1) {map = #map0} : (memref<4 × 256 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.store"(%3, %0, %arg0, %arg1) {map = #map0} : (f32, memref<4 × 256 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %4 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<4 × 256 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_at_depth1_and_reduce_slice_trip_count"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %3 = "affine.load"(%0, %arg1) {map = #map0} : (memref<100 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          %3 = "affine.load"(%0, %arg2) {map = #map0} : (memref<100 × f32>, index) -> f32
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_at_depth1_with_trip_count_20"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %3 = "affine.load"(%0, %arg1) {map = #map0} : (memref<100 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          %3 = "affine.load"(%0, %arg2) {map = #map0} : (memref<100 × f32>, index) -> f32
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_at_depth1_with_trip_count_19"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.load"(%0, %arg0) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.load"(%0, %arg0) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_with_private_memrefs_with_diff_shapes"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10 × f32>):
    %0 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.store"(%0, %arg0, %arg1) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      %1 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10 × f32>) -> (), sym_name = "should_fuse_live_out_arg_but_preserve_src_loop"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10 × f32>):
    %0 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.store"(%0, %arg0, %arg1) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      %1 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10 × f32>) -> (), sym_name = "should_fuse_live_out_arg"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%0, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"(%1) : (memref<10 × f32>) -> ()
  }) {function_type = () -> memref<10 × f32>, sym_name = "should_fuse_escaping_memref_but_preserve_src_loop"} : () -> ()
}) : () -> ()










"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × 3 × 16 × i32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          %2 = "foo"(%arg0, %arg1, %arg2) : (index, index, index) -> i32
          "affine.store"(%2, %0, %arg0, %arg1, %arg2) {map = #map0} : (i32, memref<2 × 3 × 16 × i32>, index, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.apply"(%arg0, %arg1) {map = #map5} : (index, index) -> index
        %3 = "affine.apply"(%2) {map = #map6} : (index) -> index
        %4 = "affine.load"(%0, %3, %arg1, %1) {map = #map0} : (memref<2 × 3 × 16 × i32>, index, index, index) -> i32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map7} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "R3_to_R2_reshape"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.store"(%2, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_multi_output_producer"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%4, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%3, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      %4 = "affine.load"(%2, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%4, %2, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "fusion_preventing_deps_on_middle_loop"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%4, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%2, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %2, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_and_move_to_preserve_war_dep"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %5 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    %4 = "arith.constant"() {value = 1.10000001 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %5 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%4, %2, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "fusion_preventing_dep_on_constant"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %4 = "arith.constant"() {value = 1.10000001 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %5 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %5 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%4, %2, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_and_preserve_dep_on_constant"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZloop-fusion.out.txt" str
