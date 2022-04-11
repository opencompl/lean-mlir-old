"builtin.module"() ({
^bb0:
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %2 = "arith.constant"() {value = 4 : index} : () -> index
    "affine.if"(%2) ({
      "affine.for"() ({
      ^bb0(%arg0: index):
        "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }, {
    }) {condition = #set} : (index) -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_may_execute_before_load"} : () -> ()
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
  }) {function_type = () -> (), sym_name = "dependent_loops"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    %3 = "arith.constant"() {value = 1.00000000 : f32} : () -> f32
    "affine.store"(%3, %0, %2) {map = #map} : (f32, memref<100 × f32>, index) -> ()
    %4 = "affine.load"(%1, %2) {map = #map} : (memref<100 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "different_memrefs"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.store"(%3, %0, %1) {map = #map} : (f32, memref<100 × f32>, index) -> ()
    %4 = "affine.load"(%0, %2) {map = #map} : (memref<100 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_load_different_elements"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %4 = "affine.load"(%0, %2) {map = #map} : (memref<100 × f32>, index) -> f32
    "affine.store"(%3, %0, %1) {map = #map} : (f32, memref<100 × f32>, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "load_store_different_elements"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 11 : index} : () -> index
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.store"(%2, %0, %1) {map = #map} : (f32, memref<100 × f32>, index) -> ()
    %3 = "affine.load"(%0, %1) {map = #map} : (memref<100 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_load_same_element"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 11 : index} : () -> index
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %3 = "affine.load"(%0, %1) {map = #map} : (memref<100 × f32>, index) -> f32
    %4 = "affine.load"(%0, %1) {map = #map} : (memref<100 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "load_load_same_element"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.store"(%1, %0, %arg0) {map = #map} : (f32, memref<100 × f32>, index) -> ()
    %2 = "affine.load"(%0, %arg0) {map = #map} : (memref<100 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "store_load_same_symbol"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.store"(%1, %0, %arg0) {map = #map} : (f32, memref<100 × f32>, index) -> ()
    %2 = "affine.load"(%0, %arg1) {map = #map} : (memref<100 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = (index, index) -> (), sym_name = "store_load_different_symbols"} : () -> ()
}) : () -> ()




"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 8.00000000 : f32} : () -> f32
    %3 = "affine.apply"(%1) {map = #map0} : (index) -> index
    "affine.store"(%2, %0, %3) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
    %4 = "affine.apply"(%1) {map = #map1} : (index) -> index
    %5 = "affine.load"(%0, %4) {map = #map0} : (memref<100 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_load_diff_element_affine_apply_const"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %2 = "arith.constant"() {value = 9 : index} : () -> index
    %3 = "arith.constant"() {value = 11 : index} : () -> index
    %4 = "affine.apply"(%2) {map = #map0} : (index) -> index
    "affine.store"(%1, %0, %4) {map = #map1} : (f32, memref<100 × f32>, index) -> ()
    %5 = "affine.apply"(%3) {map = #map2} : (index) -> index
    %6 = "affine.load"(%0, %5) {map = #map1} : (memref<100 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_load_same_element_affine_apply_const"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %2 = "affine.apply"(%arg0) {map = #map} : (index) -> index
    "affine.store"(%1, %0, %2) {map = #map} : (f32, memref<100 × f32>, index) -> ()
    %3 = "affine.apply"(%arg0) {map = #map} : (index) -> index
    %4 = "affine.load"(%0, %3) {map = #map} : (memref<100 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "store_load_affine_apply_symbol"} : () -> ()
}) : () -> ()




"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %2 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
    "affine.store"(%1, %0, %2) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
    %3 = "affine.apply"(%arg0) {map = #map1} : (index) -> index
    %4 = "affine.load"(%0, %3) {map = #map0} : (memref<100 × f32>, index) -> f32
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "store_load_affine_apply_symbol_offset"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %2 = "arith.constant"() {value = 10 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %3) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      %4 = "affine.apply"(%2) {map = #map0} : (index) -> index
      %5 = "affine.load"(%0, %4) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_range_load_after_range"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %2 = "arith.constant"() {value = 10 : index} : () -> index
    "affine.for"(%arg1) ({
    ^bb0(%arg2: index):
      %3 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %3) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      %4 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      %5 = "affine.load"(%0, %4) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index) -> (), sym_name = "store_load_func_symbol"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %2 = "arith.constant"() {value = 10 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %3) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      %4 = "affine.apply"(%2) {map = #map1} : (index) -> index
      %5 = "affine.load"(%0, %4) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_range_load_last_in_range"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %3) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      %4 = "affine.apply"(%2) {map = #map0} : (index) -> index
      %5 = "affine.load"(%0, %4) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_range_load_before_range"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %3) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      %4 = "affine.apply"(%2) {map = #map1} : (index) -> index
      %5 = "affine.load"(%0, %4) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_range_load_first_in_range"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %2) {map = #map1} : (f32, memref<100 × f32>, index) -> ()
      %3 = "affine.apply"(%arg0) {map = #map1} : (index) -> index
      %4 = "affine.load"(%0, %3) {map = #map1} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "store_plus_3"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %2) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      %3 = "affine.apply"(%arg0) {map = #map1} : (index) -> index
      %4 = "affine.load"(%0, %3) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "load_minus_2"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.apply"(%arg0, %arg1) {map = #map0} : (index, index) -> index
        %3 = "affine.apply"(%arg0, %arg1) {map = #map1} : (index, index) -> index
        "affine.store"(%1, %0, %2, %3) {map = #map2} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        %4 = "affine.apply"(%arg0, %arg1) {map = #map0} : (index, index) -> index
        %5 = "affine.apply"(%arg0, %arg1) {map = #map1} : (index, index) -> index
        %6 = "affine.load"(%0, %4, %5) {map = #map2} : (memref<10 × 10 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map3, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map3, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "perfectly_nested_loops_loop_independent"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.apply"(%arg0, %arg1) {map = #map0} : (index, index) -> index
        %3 = "affine.apply"(%arg0, %arg1) {map = #map1} : (index, index) -> index
        "affine.store"(%1, %0, %2, %3) {map = #map2} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        %4 = "affine.apply"(%arg0, %arg1) {map = #map3} : (index, index) -> index
        %5 = "affine.apply"(%arg0, %arg1) {map = #map1} : (index, index) -> index
        %6 = "affine.load"(%0, %4, %5) {map = #map2} : (memref<10 × 10 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "perfectly_nested_loops_loop_carried_at_depth1"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.apply"(%arg0, %arg1) {map = #map0} : (index, index) -> index
        %3 = "affine.apply"(%arg0, %arg1) {map = #map1} : (index, index) -> index
        "affine.store"(%1, %0, %2, %3) {map = #map2} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        %4 = "affine.apply"(%arg0, %arg1) {map = #map0} : (index, index) -> index
        %5 = "affine.apply"(%arg0, %arg1) {map = #map3} : (index, index) -> index
        %6 = "affine.load"(%0, %4, %5) {map = #map2} : (memref<10 × 10 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "perfectly_nested_loops_loop_carried_at_depth2"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.apply"(%arg0, %arg1) {map = #map0} : (index, index) -> index
        %3 = "affine.apply"(%arg0, %arg1) {map = #map1} : (index, index) -> index
        "affine.store"(%1, %0, %2, %3) {map = #map2} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map3, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.apply"(%arg0, %arg1) {map = #map0} : (index, index) -> index
        %3 = "affine.apply"(%arg0, %arg1) {map = #map1} : (index, index) -> index
        %4 = "affine.load"(%0, %2, %3) {map = #map2} : (memref<10 × 10 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map3, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map3, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "one_common_loop"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      %3 = "affine.load"(%0, %2) {map = #map0} : (memref<100 × f32>, index) -> f32
      %4 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%3, %1, %4) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      %5 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      %6 = "affine.load"(%1, %5) {map = #map0} : (memref<100 × f32>, index) -> f32
      %7 = "affine.apply"(%arg0) {map = #map1} : (index) -> index
      "affine.store"(%6, %0, %7) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dependence_cycle"} : () -> ()
}) : () -> ()









"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"(%arg0) ({
    ^bb0(%arg2: index):
      "affine.for"(%arg1) ({
      ^bb0(%arg3: index):
        %2 = "affine.apply"(%arg2, %arg3) {map = #map0} : (index, index) -> index
        %3 = "affine.apply"(%arg2, %arg3) {map = #map1} : (index, index) -> index
        %4 = "affine.load"(%0, %2, %3) {map = #map2} : (memref<10 × 10 × f32>, index, index) -> f32
        %5 = "affine.apply"(%arg2, %arg3) {map = #map3} : (index, index) -> index
        %6 = "affine.apply"(%arg2, %arg3) {map = #map4} : (index, index) -> index
        "affine.store"(%1, %0, %5, %6) {map = #map2} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map5, step = 1 : index, upper_bound = #map6} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map5, step = 1 : index, upper_bound = #map6} : (index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index) -> (), sym_name = "negative_and_positive_direction_vectors"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.apply"(%arg1) {map = #map0} : (index) -> index
        %3 = "affine.load"(%0, %2) {map = #map1} : (memref<100 × f32>, index) -> f32
        %4 = "affine.apply"(%arg1) {map = #map1} : (index) -> index
        "affine.store"(%1, %0, %4) {map = #map1} : (f32, memref<100 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "war_raw_waw_deps"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      %3 = "affine.load"(%0, %2) {map = #map1} : (memref<100 × f32>, index) -> f32
      %4 = "affine.apply"(%arg0) {map = #map2} : (index) -> index
      "affine.store"(%1, %0, %4) {map = #map1} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map3, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "mod_deps"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × 100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.store"(%1, %0, %arg0, %arg1) {map = #map0} : (f32, memref<100 × 100 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          "affine.for"() ({
          ^bb0(%arg3: index):
            %2 = "affine.apply"(%arg2, %arg3) {map = #map4} : (index, index) -> index
            %3 = "affine.load"(%0, %2, %arg1) {map = #map0} : (memref<100 × 100 × f32>, index, index) -> f32
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "loop_nest_depth"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × 2 × 2 × i32>
    %1 = "arith.constant"() {value = 0 : i32} : () -> i32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          %2 = "affine.apply"(%arg0, %arg1, %arg2) {map = #map0} : (index, index, index) -> index
          %3 = "affine.apply"(%arg0, %arg1, %arg2) {map = #map1} : (index, index, index) -> index
          %4 = "affine.apply"(%arg0, %arg1, %arg2) {map = #map2} : (index, index, index) -> index
          "affine.store"(%1, %0, %2, %3, %4) {map = #map3} : (i32, memref<2 × 2 × 2 × i32>, index, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "mod_div_3d"} : () -> ()
}) : () -> ()


















"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 0 : i32} : () -> i32
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2x2x3x3x16x1xi32>
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 9 × i32>
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
                "affine.store"(%1, %2, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {map = #map0} : (i32, memref<2x2x3x3x16x1xi32>, index, index, index, index, index, index) -> ()
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
        %4 = "affine.apply"(%arg0, %arg1) {map = #map6} : (index, index) -> index
        %5 = "affine.apply"(%4) {map = #map7} : (index) -> index
        %6 = "affine.apply"(%4) {map = #map8} : (index) -> index
        %7 = "affine.apply"(%4) {map = #map9} : (index) -> index
        %8 = "affine.apply"(%4) {map = #map10} : (index) -> index
        %9 = "affine.apply"(%4) {map = #map11} : (index) -> index
        %10 = "affine.apply"(%4) {map = #map12} : (index) -> index
        %11 = "affine.load"(%2, %5, %6, %8, %9, %7, %10) {map = #map0} : (memref<2x2x3x3x16x1xi32>, index, index, index, index, index, index) -> i32
        "affine.store"(%11, %3, %arg0, %arg1) {map = #map13} : (i32, memref<64 × 9 × i32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map14} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map15} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "delinearize_mod_floordiv"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      %2 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 2 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "strided_loop_with_dependence_at_depth2"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %2) {map = #map1} : (f32, memref<10 × f32>, index) -> ()
      %3 = "affine.load"(%0, %arg0) {map = #map1} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 2 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "strided_loop_with_no_dependence"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %2) {map = #map1} : (f32, memref<10 × f32>, index) -> ()
      %3 = "affine.load"(%0, %arg0) {map = #map1} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 2 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "strided_loop_with_loop_carried_dependence_at_depth1"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %2) {map = #map1} : (f32, memref<100 × f32>, index) -> ()
      "affine.for"(%arg0, %arg0) ({
      ^bb0(%arg1: index):
        %3 = "affine.load"(%0, %arg1) {map = #map1} : (memref<100 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map3, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_dep_store_depth1_load_depth2"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"(%arg0, %arg0) ({
      ^bb0(%arg1: index):
        "affine.store"(%1, %0, %arg1) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : (index, index) -> ()
      %2 = "affine.apply"(%arg0) {map = #map2} : (index) -> index
      %3 = "affine.load"(%0, %2) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map3, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_dep_store_depth2_load_depth1"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.if"(%arg0) ({
        %2 = "affine.load"(%0, %arg0) {map = #map0} : (memref<100 × f32>, index) -> f32
        "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_affine_for_if_same_block"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.if"(%arg0) ({
        %2 = "affine.load"(%0, %arg0) {map = #map0} : (memref<100 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }, {
        "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {condition = #set} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_affine_for_if_separated"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.if"(%arg0) ({
        %2 = "affine.load"(%0, %arg0) {map = #map0} : (memref<100 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set0} : (index) -> ()
      "affine.if"(%arg0) ({
        "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set1} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_affine_for_if_partially_joined"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × 100 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.if"(%arg0) ({
        "affine.for"() ({
        ^bb0(%arg1: index):
          %2 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<100 × 100 × f32>, index, index) -> f32
          "affine.if"(%arg0, %arg1) ({
            "affine.store"(%1, %0, %arg0, %arg1) {map = #map0} : (f32, memref<100 × 100 × f32>, index, index) -> ()
            "affine.yield"() : () -> ()
          }, {
          }) {condition = #set0} : (index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set1} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_interleaved_affine_for_if"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<101 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "memref.dim"(%0, %1) : (memref<101 × f32>, index) -> index
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.if"(%arg0, %2) ({
        %4 = "affine.load"(%0, %arg0) {map = #map0} : (memref<101 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set0} : (index, index) -> ()
      "affine.if"(%arg0) ({
        "affine.store"(%3, %0, %arg0) {map = #map0} : (f32, memref<101 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set1} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_interleaved_affine_for_if"} : () -> ()
}) : () -> ()


