"builtin.module"() ({
^bb0:
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (16)>
#map3 = affine_map<() -> (5)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
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
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "slice_depth1_loop_nest"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (5)>
#map3 = affine_map<() -> (16)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
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
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "forward_slice_slice_depth1_loop_nest"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0 + 2)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (16)>
#map4 = affine_map<(d0) -> (d0 - 1)>
#map5 = affine_map<() -> (4)>
#map6 = affine_map<() -> (8)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.store"(%1, %0, %2) {map = #map1} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map4} : (index) -> index
      %3 = "affine.load"(%0, %2) {map = #map1} : (memref<100 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map5, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "slice_depth1_loop_nest_with_offsets"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (16)>
#map3 = affine_map<() -> (8)>
#map4 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × 100 × f32>
    %1 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.store"(%1, %0, %arg0, %arg1) {map = #map0} : (f32, memref<100 × 100 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<100 × 100 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "slice_depth2_loop_nest"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (16)>
#map3 = affine_map<() -> (8)>
#map4 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × 100 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.store"(%2, %0, %arg0, %arg1) {map = #map0} : (f32, memref<100 × 100 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %4 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<100 × 100 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      %3 = "affine.load"(%0, %arg0, %1) {map = #map0} : (memref<100 × 100 × f32>, index, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "slice_depth2_loop_nest_two_loads"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (16)>
#map3 = affine_map<() -> (8)>
#map4 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × 100 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.store"(%2, %0, %arg0, %arg1) {map = #map0} : (f32, memref<100 × 100 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.store"(%2, %0, %arg0, %1) {map = #map0} : (f32, memref<100 × 100 × f32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %3 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<100 × 100 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "slice_depth2_loop_nest_two_stores"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (16)>
#map3 = affine_map<() -> (10)>
#map4 = affine_map<() -> (8)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × 100 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.store"(%2, %0, %arg0, %arg1) {map = #map0} : (f32, memref<100 × 100 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %3 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<100 × 100 × f32>, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "slice_loop_nest_with_smaller_outer_trip_count"} : () -> ()
}) : () -> ()

// -----
