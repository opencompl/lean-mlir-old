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
      "prevent.dce"(%2) : (f32) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "slice_depth1_loop_nest"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10x10xf32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
        %5 = "affine.load"(%0, %arg0, %arg1) {map = #map1} : (memref<10x10xf32>, index, index) -> f32
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

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
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
      "prevent.dce"(%4) : (f32) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%4, %2, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_avoiding_dependence_cycle"} : () -> ()
}) : () -> ()

// -----
