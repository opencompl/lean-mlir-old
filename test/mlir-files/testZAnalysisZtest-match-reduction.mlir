#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (0)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<?xf32>, %arg1: tensor<1xf32>):
    %0 = "linalg.generic"(%arg0, %arg1) ({
    ^bb0(%arg2: f32, %arg3: f32):
      %1 = "arith.addf"(%arg2, %arg3) : (f32, f32) -> f32
      "linalg.yield"(%1) : (f32) -> ()
    }) {indexing_maps = [#map0, #map1], iterator_types = ["reduction"], operand_segment_sizes = dense<1> : vector<2xi32>} : (tensor<?xf32>, tensor<1xf32>) -> tensor<1xf32>
    "func.return"() : () -> ()
  }) {function_type = (tensor<?xf32>, tensor<1xf32>) -> (), sym_name = "linalg_red_add"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (512)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<() -> (256)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<256x512xf32>, %arg1: memref<256xf32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      %1 = "affine.for"(%0) ({
      ^bb0(%arg3: index, %arg4: f32):
        %2 = "affine.load"(%arg0, %arg2, %arg3) {map = #map0} : (memref<256x512xf32>, index, index) -> f32
        %3 = "arith.addf"(%arg4, %2) : (f32, f32) -> f32
        "affine.yield"(%3) : (f32) -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (f32) -> f32
      "affine.store"(%1, %arg1, %arg2) {map = #map3} : (f32, memref<256xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<256x512xf32>, memref<256xf32>) -> (), sym_name = "affine_red_add"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>):
    %0 = "linalg.generic"(%arg0, %arg1) ({
    ^bb0(%arg2: f32, %arg3: f32):
      %1 = "arith.cmpf"(%arg2, %arg3) {predicate = 2 : i64} : (f32, f32) -> i1
      %2 = "arith.select"(%1, %arg2, %arg3) : (i1, f32, f32) -> f32
      "linalg.yield"(%2) : (f32) -> ()
    }) {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"], operand_segment_sizes = dense<1> : vector<2xi32>} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "func.return"() : () -> ()
  }) {function_type = (tensor<4x4xf32>, tensor<4xf32>) -> (), sym_name = "linalg_red_max"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>):
    %0 = "linalg.generic"(%arg0, %arg1) ({
    ^bb0(%arg2: f32, %arg3: f32):
      %1 = "arith.mulf"(%arg2, %arg2) : (f32, f32) -> f32
      %2 = "arith.subf"(%1, %arg2) : (f32, f32) -> f32
      %3 = "arith.addf"(%2, %arg3) : (f32, f32) -> f32
      "linalg.yield"(%3) : (f32) -> ()
    }) {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"], operand_segment_sizes = dense<1> : vector<2xi32>} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "func.return"() : () -> ()
  }) {function_type = (tensor<4x4xf32>, tensor<4xf32>) -> (), sym_name = "linalg_fused_red_add"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (512)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<512xf32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %1 = "affine.for"(%0) ({
    ^bb0(%arg1: index, %arg2: f32):
      %2 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<512xf32>, index) -> f32
      %3 = "arith.addf"(%2, %arg2) : (f32, f32) -> f32
      "affine.yield"(%2) : (f32) -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (f32) -> f32
    "func.return"() : () -> ()
  }) {function_type = (memref<512xf32>) -> (), sym_name = "affine_no_red_rec"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (512)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<512xf32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %1:2 = "affine.for"(%0, %0) ({
    ^bb0(%arg1: index, %arg2: f32, %arg3: f32):
      %2 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<512xf32>, index) -> f32
      %3 = "arith.addf"(%arg3, %arg2) : (f32, f32) -> f32
      "affine.yield"(%3, %2) : (f32, f32) -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (f32, f32) -> (f32, f32)
    "func.return"() : () -> ()
  }) {function_type = (memref<512xf32>) -> (), sym_name = "affine_output_dep"} : () -> ()
}) : () -> ()

// -----
