"builtin.module"() ({
^bb0:
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (3)>
#map3 = affine_map<() -> (4)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<3x4xf32>, %arg1: memref<4x3xf32>, %arg2: memref<3x3xf32>, %arg3: memref<3x3xf32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<3x3xf32>
    "affine.for"() ({
    ^bb0(%arg4: index):
      "affine.for"() ({
      ^bb0(%arg5: index):
        "affine.store"(%0, %1, %arg4, %arg5) {map = #map0} : (f32, memref<3x3xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg4: index):
      "affine.for"() ({
      ^bb0(%arg5: index):
        "affine.for"() ({
        ^bb0(%arg6: index):
          %2 = "affine.load"(%arg1, %arg6, %arg5) {map = #map0} : (memref<4x3xf32>, index, index) -> f32
          %3 = "affine.load"(%arg0, %arg4, %arg6) {map = #map0} : (memref<3x4xf32>, index, index) -> f32
          %4 = "arith.mulf"(%3, %2) : (f32, f32) -> f32
          %5 = "affine.load"(%1, %arg4, %arg5) {map = #map0} : (memref<3x3xf32>, index, index) -> f32
          %6 = "arith.addf"(%5, %4) : (f32, f32) -> f32
          "affine.store"(%6, %1, %arg4, %arg5) {map = #map0} : (f32, memref<3x3xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg4: index):
      "affine.for"() ({
      ^bb0(%arg5: index):
        %2 = "affine.load"(%arg2, %arg4, %arg5) {map = #map0} : (memref<3x3xf32>, index, index) -> f32
        %3 = "affine.load"(%1, %arg4, %arg5) {map = #map0} : (memref<3x3xf32>, index, index) -> f32
        %4 = "arith.addf"(%3, %2) : (f32, f32) -> f32
        "affine.store"(%4, %arg3, %arg4, %arg5) {map = #map0} : (f32, memref<3x3xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<3x4xf32>, memref<4x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> (), sym_name = "mul_add_0"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (1)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1xf32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.store"(%0, %arg0, %arg1) {map = #map0} : (f32, memref<1xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      %1 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<1xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      %1 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<1xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1xf32>) -> (), sym_name = "should_fuse_multi_outgoing_edge_store_producer"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)>
#map5 = affine_map<() -> (1024)>
#map6 = affine_map<() -> (64)>
#map7 = affine_map<() -> (16)>
#map8 = affine_map<(d0, d1) -> (d0, d1)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1xf32>, %arg1: memref<1xf32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      %1 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<1xf32>, index) -> f32
      "affine.store"(%0, %arg1, %arg2) {map = #map0} : (f32, memref<1xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.store"(%0, %arg0, %arg2) {map = #map0} : (f32, memref<1xf32>, index) -> ()
      %1 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<1xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1xf32>, memref<1xf32>) -> (), sym_name = "should_fuse_producer_with_multi_outgoing_edges"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<1024x1024xf32>, %arg1: memref<16x64x1024xf32>, %arg2: memref<1024x1024xf32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<1024x1024xf32>
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        "affine.for"() ({
        ^bb0(%arg5: index):
          %1 = "affine.load"(%arg1, %arg3, %arg4, %arg5) {map = #map3} : (memref<16x64x1024xf32>, index, index, index) -> f32
          "affine.store"(%1, %0, %arg3, %arg4, %arg5) {map = #map4} : (f32, memref<1024x1024xf32>, index, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map6} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map7} : () -> ()
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        "affine.for"() ({
        ^bb0(%arg5: index):
          %1 = "affine.load"(%0, %arg5, %arg4) {map = #map8} : (memref<1024x1024xf32>, index, index) -> f32
          %2 = "affine.load"(%arg0, %arg3, %arg5) {map = #map8} : (memref<1024x1024xf32>, index, index) -> f32
          %3 = "arith.mulf"(%2, %1) : (f32, f32) -> f32
          %4 = "affine.load"(%arg2, %arg3, %arg4) {map = #map8} : (memref<1024x1024xf32>, index, index) -> f32
          %5 = "arith.addf"(%4, %3) : (f32, f32) -> f32
          "affine.store"(%5, %arg2, %arg3, %arg4) {map = #map8} : (f32, memref<1024x1024xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1024x1024xf32>, memref<16x64x1024xf32>, memref<1024x1024xf32>) -> (), sym_name = "reshape_into_matmul"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0, d1 * 4)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (5)>
#map3 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>, %arg2: memref<10x20xf32>):
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        %0 = "affine.vector_load"(%arg0, %arg3, %arg4) {map = #map0} : (memref<10x20xf32>, index, index) -> vector<4xf32>
        "affine.vector_store"(%0, %arg1, %arg3, %arg4) {map = #map0} : (vector<4xf32>, memref<10x20xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        %0 = "affine.vector_load"(%arg1, %arg3, %arg4) {map = #map0} : (memref<10x20xf32>, index, index) -> vector<4xf32>
        "affine.vector_store"(%0, %arg2, %arg3, %arg4) {map = #map0} : (vector<4xf32>, memref<10x20xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10x20xf32>, memref<10x20xf32>, memref<10x20xf32>) -> (), sym_name = "vector_loop"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<32xf32>, %arg1: memref<32xf32>):
    "affine.for"() ({
    ^bb0(%arg2: index):
      %0 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %1 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %2 = "arith.addf"(%0, %1) : (f32, f32) -> f32
      "affine.store"(%2, %arg0, %arg2) {map = #map0} : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      %0 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %1 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %2 = "arith.subf"(%0, %1) : (f32, f32) -> f32
      "affine.store"(%2, %arg0, %arg2) {map = #map0} : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      %0 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %1 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %2 = "arith.mulf"(%0, %1) : (f32, f32) -> f32
      "affine.store"(%2, %arg0, %arg2) {map = #map0} : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      %0 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %1 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %2 = "arith.divf"(%0, %1) : (f32, f32) -> f32
      "affine.store"(%2, %arg0, %arg2) {map = #map0} : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<32xf32>, memref<32xf32>) -> (), sym_name = "multi_outgoing_edges"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (1)>
#map2 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: index):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "memref.alloc"(%arg3) {operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (index) -> memref<?xf32>
    "affine.for"() ({
    ^bb0(%arg4: index):
      %2 = "affine.load"(%arg0, %arg4) {map = #map0} : (memref<?xf32>, index) -> f32
      %3 = "affine.load"(%arg1, %arg4) {map = #map0} : (memref<?xf32>, index) -> f32
      %4 = "arith.addf"(%2, %3) : (f32, f32) -> f32
      "affine.store"(%4, %1, %arg4) {map = #map0} : (f32, memref<?xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg4: index):
      %2 = "affine.load"(%1, %arg4) {map = #map0} : (memref<?xf32>, index) -> f32
      %3 = "affine.load"(%arg1, %arg4) {map = #map0} : (memref<?xf32>, index) -> f32
      %4 = "arith.mulf"(%2, %3) : (f32, f32) -> f32
      "affine.store"(%4, %arg2, %arg4) {map = #map0} : (f32, memref<?xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<?xf32>, memref<?xf32>, memref<?xf32>, index) -> (), sym_name = "calc"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<32xf32>, %arg1: memref<32xf32>):
    "affine.for"() ({
    ^bb0(%arg2: index):
      %0 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %1 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %2 = "arith.addf"(%0, %1) : (f32, f32) -> f32
      "affine.store"(%2, %arg0, %arg2) {map = #map0} : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      %0 = "memref.load"(%arg0, %arg2) : (memref<32xf32>, index) -> f32
      %1 = "memref.load"(%arg1, %arg2) : (memref<32xf32>, index) -> f32
      %2 = "arith.subf"(%0, %1) : (f32, f32) -> f32
      "memref.store"(%2, %arg0, %arg2) : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      %0 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %1 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %2 = "arith.mulf"(%0, %1) : (f32, f32) -> f32
      "affine.store"(%2, %arg0, %arg2) {map = #map0} : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<32xf32>, memref<32xf32>) -> (), sym_name = "should_not_fuse_since_non_affine_users"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<32xf32>, %arg1: memref<32xf32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<f32>
    "affine.for"() ({
    ^bb0(%arg2: index):
      %2 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %3 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %4 = "arith.addf"(%2, %3) : (f32, f32) -> f32
      "memref.store"(%4, %0) : (f32, memref<f32>) -> ()
      "affine.store"(%4, %arg0, %arg2) {map = #map0} : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    %1 = "memref.load"(%0) : (memref<f32>) -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      %2 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %3 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %4 = "arith.mulf"(%2, %3) : (f32, f32) -> f32
      %5 = "arith.subf"(%4, %1) : (f32, f32) -> f32
      "affine.store"(%5, %arg0, %arg2) {map = #map0} : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "memref.dealloc"(%0) : (memref<f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<32xf32>, memref<32xf32>) -> (), sym_name = "should_not_fuse_since_top_level_non_affine_users"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<32xf32>, %arg1: memref<32xf32>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      %2 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %3 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %4 = "arith.addf"(%2, %3) : (f32, f32) -> f32
      "affine.store"(%4, %arg0, %arg2) {map = #map0} : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "memref.store"(%1, %arg0, %0) : (f32, memref<32xf32>, index) -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      %2 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %3 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<32xf32>, index) -> f32
      %4 = "arith.addf"(%2, %3) : (f32, f32) -> f32
      "affine.store"(%4, %arg0, %arg2) {map = #map0} : (f32, memref<32xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<32xf32>, memref<32xf32>) -> (), sym_name = "should_not_fuse_since_top_level_non_affine_mem_write_users"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (128)>
#map3 = affine_map<(d0) -> (d0 mod 128)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<() -> (512)>
#map6 = affine_map<() -> (20)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<128xf32>, %arg1: memref<20x512xf32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<128xf32>
    "affine.for"() ({
    ^bb0(%arg2: index):
      %1 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<128xf32>, index) -> f32
      "affine.store"(%1, %0, %arg2) {map = #map0} : (f32, memref<128xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.for"() ({
      ^bb0(%arg3: index):
        %1 = "affine.load"(%0, %arg3) {map = #map3} : (memref<128xf32>, index) -> f32
        "affine.store"(%1, %arg1, %arg2, %arg3) {map = #map4} : (f32, memref<20x512xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<128xf32>, memref<20x512xf32>) -> (), sym_name = "fuse_minor_affine_map"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<10xf32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<10xf32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<10xf32>
    %3 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%3, %0, %arg0) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.store"(%3, %2, %arg0) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      %4 = "affine.load"(%2, %arg0) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_multi_store_producer_and_privatize_memfefs"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<10xf32>, %arg1: memref<10xf32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.store"(%0, %arg0, %arg2) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.store"(%0, %arg1, %arg2) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      %1 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      %1 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10xf32>, memref<10xf32>) -> (), sym_name = "should_fuse_multi_store_producer_with_escaping_memrefs_and_remove_src"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
#map3 = affine_map<() -> (5)>
#map4 = affine_map<() -> (16)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10xf32>, %arg1: memref<10xf32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.store"(%0, %arg0, %arg2) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.store"(%0, %arg1, %arg2) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      %1 = "affine.load"(%arg0, %arg2) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      %1 = "affine.load"(%arg1, %arg2) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10xf32>, memref<10xf32>) -> (), sym_name = "should_fuse_multi_store_producer_with_escaping_memrefs_and_preserve_src"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<16xf32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<16xf32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<16xf32>
    %2 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      %4 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<16xf32>, index) -> f32
      "affine.store"(%4, %0, %arg1) {map = #map0} : (f32, memref<16xf32>, index) -> ()
      "affine.store"(%4, %1, %arg1) {map = #map0} : (f32, memref<16xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "memref.dealloc"(%1) : (memref<16xf32>) -> ()
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<16xf32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      %4 = "affine.load"(%0, %arg1) {map = #map0} : (memref<16xf32>, index) -> f32
      %5 = "arith.addf"(%2, %4) : (f32, f32) -> f32
      "affine.store"(%5, %3, %arg1) {map = #map0} : (f32, memref<16xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
    "memref.dealloc"(%0) : (memref<16xf32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<16xf32>) -> (), sym_name = "should_not_fuse_due_to_dealloc"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10xf32>, %arg1: memref<f32>):
    "affine.for"() ({
    ^bb0(%arg2: index):
      %1 = "affine.load"(%arg1) {map = #map0} : (memref<f32>) -> f32
      "affine.store"(%1, %arg0, %arg2) {map = #map1} : (f32, memref<10xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    %0 = "affine.load"(%arg1) {map = #map0} : (memref<f32>) -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      %1 = "affine.load"(%arg0, %arg2) {map = #map1} : (memref<10xf32>, index) -> f32
      %2 = "arith.divf"(%0, %1) : (f32, f32) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10xf32>, memref<f32>) -> (), sym_name = "should_fuse_defining_node_has_no_dependence_from_source_node"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10xf32>, %arg1: memref<f32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.store"(%0, %arg1) {map = #map0} : (f32, memref<f32>) -> ()
      "affine.store"(%0, %arg0, %arg2) {map = #map1} : (f32, memref<10xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    %1 = "affine.load"(%arg1) {map = #map0} : (memref<f32>) -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      %2 = "affine.load"(%arg0, %arg2) {map = #map1} : (memref<10xf32>, index) -> f32
      %3 = "arith.divf"(%1, %2) : (f32, f32) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10xf32>, memref<f32>) -> (), sym_name = "should_not_fuse_defining_node_has_dependence_from_source_loop"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
#map3 = affine_map<() -> ()>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10xf32>, %arg1: memref<10xf32>, %arg2: memref<f32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.store"(%0, %arg0, %arg3) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.store"(%0, %arg1, %arg3) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg3: index):
      %2 = "affine.load"(%arg1, %arg3) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.store"(%2, %arg2) {map = #map3} : (f32, memref<f32>) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    %1 = "affine.load"(%arg2) {map = #map3} : (memref<f32>) -> f32
    "affine.for"() ({
    ^bb0(%arg3: index):
      %2 = "affine.load"(%arg0, %arg3) {map = #map0} : (memref<10xf32>, index) -> f32
      %3 = "arith.divf"(%1, %2) : (f32, f32) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10xf32>, memref<10xf32>, memref<f32>) -> (), sym_name = "should_not_fuse_defining_node_has_transitive_dependence_from_source_loop"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10xf32>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.store"(%0, %arg0, %arg1) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    %1 = "affine.for"(%0) ({
    ^bb0(%arg1: index, %arg2: f32):
      %2 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"(%2) : (f32) -> ()
    }) {lower_bound = #map1, step = 2 : index, upper_bound = #map2} : (f32) -> f32
    "func.return"() : () -> ()
  }) {function_type = (memref<10xf32>) -> (), sym_name = "should_not_fuse_dest_loop_nest_return_value"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10xf32>):
    %0 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %1 = "affine.for"(%0) ({
    ^bb0(%arg1: index, %arg2: f32):
      %2 = "arith.addf"(%arg2, %arg2) : (f32, f32) -> f32
      "affine.store"(%2, %arg0, %arg1) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.yield"(%2) : (f32) -> ()
    }) {lower_bound = #map1, step = 2 : index, upper_bound = #map2} : (f32) -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      %2 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10xf32>) -> (), sym_name = "should_not_fuse_src_loop_nest_return_value"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (16)>
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (memref<16xf32>) -> (), sym_name = "some_function", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<16xf32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<16xf32>
    %1 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      %3 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<16xf32>, index) -> f32
      "affine.store"(%3, %0, %arg1) {map = #map0} : (f32, memref<16xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.call"(%0) {callee = @some_function} : (memref<16xf32>) -> ()
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<16xf32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      %3 = "affine.load"(%0, %arg1) {map = #map0} : (memref<16xf32>, index) -> f32
      %4 = "arith.addf"(%1, %3) : (f32, f32) -> f32
      "affine.store"(%4, %2, %arg1) {map = #map0} : (f32, memref<16xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<16xf32>) -> (), sym_name = "call_op_prevents_fusion"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (16)>
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "some_function", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<16xf32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<16xf32>
    %1 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      %3 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<16xf32>, index) -> f32
      "affine.store"(%3, %0, %arg1) {map = #map0} : (f32, memref<16xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.call"() {callee = @some_function} : () -> ()
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<16xf32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      %3 = "affine.load"(%0, %arg1) {map = #map0} : (memref<16xf32>, index) -> f32
      %4 = "arith.addf"(%1, %3) : (f32, f32) -> f32
      "affine.store"(%4, %2, %arg1) {map = #map0} : (f32, memref<16xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<16xf32>) -> (), sym_name = "call_op_does_not_prevent_fusion"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
#map3 = affine_map<() -> (7)>
#map4 = affine_map<() -> (5)>
#map5 = affine_map<() -> (9)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10xf32>):
    %0 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.store"(%0, %arg0, %arg1) {map = #map0} : (f32, memref<10xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      %1 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      %1 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<10xf32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10xf32>) -> (), sym_name = "should_fuse_with_both_consumers_separately"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (5)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<() -> (64)>
#map6 = affine_map<() -> (1)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<5xf32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<6xf32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<5xf32>
    %3 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%0, %arg0) {map = #map0} : (memref<5xf32>, index) -> f32
      "affine.store"(%4, %1, %arg0) {map = #map1} : (f32, memref<6xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %4 = "affine.load"(%1, %arg0) {map = #map0} : (memref<6xf32>, index) -> f32
      %5 = "arith.mulf"(%4, %3) : (f32, f32) -> f32
      "affine.store"(%5, %2, %arg0) {map = #map0} : (f32, memref<5xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "no_fusion_cannot_compute_valid_slice"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<64x64xf32, 1>, %arg1: memref<1x64xf32, 1>, %arg2: memref<1x64xf32, 1>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %1 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %2 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<f32, 1>
    %3 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<f32, 1>
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        %4 = "affine.for"(%0) ({
        ^bb0(%arg5: index, %arg6: f32):
          %6 = "affine.load"(%arg0, %arg5, %arg4) {map = #map4} : (memref<64x64xf32, 1>, index, index) -> f32
          %7 = "arith.addf"(%arg6, %6) : (f32, f32) -> f32
          "affine.yield"(%7) : (f32) -> ()
        }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : (f32) -> f32
        %5 = "arith.addf"(%4, %4) : (f32, f32) -> f32
        "affine.store"(%5, %arg1, %arg3, %arg4) {map = #map4} : (f32, memref<1x64xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map6} : () -> ()
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        %4 = "affine.for"(%1) ({
        ^bb0(%arg5: index, %arg6: f32):
          %6 = "affine.load"(%arg0, %arg5, %arg4) {map = #map4} : (memref<64x64xf32, 1>, index, index) -> f32
          %7 = "arith.mulf"(%arg6, %6) : (f32, f32) -> f32
          "affine.yield"(%7) : (f32) -> ()
        }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : (f32) -> f32
        %5 = "arith.mulf"(%4, %4) : (f32, f32) -> f32
        "affine.store"(%5, %arg2, %arg3, %arg4) {map = #map4} : (f32, memref<1x64xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<64x64xf32, 1>, memref<1x64xf32, 1>, memref<1x64xf32, 1>) -> (), sym_name = "reduce_add_f32_f32"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (64)>
#map3 = affine_map<() -> (1)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<64x64xf32, 1>, %arg1: memref<1x64xf32, 1>, %arg2: memref<1x64xf32, 1>):
    %0 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %1 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %2 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<f32, 1>
    %3 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<f32, 1>
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        %4 = "affine.for"(%0) ({
        ^bb0(%arg5: index, %arg6: f32):
          %6 = "affine.load"(%arg0, %arg5, %arg4) {map = #map0} : (memref<64x64xf32, 1>, index, index) -> f32
          %7 = "arith.addf"(%arg6, %6) : (f32, f32) -> f32
          "affine.yield"(%7) : (f32) -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (f32) -> f32
        %5 = "arith.addf"(%4, %4) : (f32, f32) -> f32
        "affine.store"(%5, %arg1, %arg3, %arg4) {map = #map0} : (f32, memref<1x64xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        %4 = "affine.for"(%1) ({
        ^bb0(%arg5: index, %arg6: f32):
          %6 = "affine.load"(%arg0, %arg5, %arg4) {map = #map0} : (memref<64x64xf32, 1>, index, index) -> f32
          %7 = "arith.mulf"(%arg6, %6) : (f32, f32) -> f32
          "affine.yield"(%7) : (f32) -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (f32) -> f32
        %5 = "arith.mulf"(%4, %4) : (f32, f32) -> f32
        "affine.store"(%5, %arg2, %arg3, %arg4) {map = #map0} : (f32, memref<1x64xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<64x64xf32, 1>, memref<1x64xf32, 1>, memref<1x64xf32, 1>) -> (), sym_name = "reduce_add_non_innermost"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (10)>
#map4 = affine_map<() -> (20)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<20x10xf32, 1>, %arg1: memref<20x10xf32, 1>, %arg2: memref<20x10xf32, 1>, %arg3: memref<20x10xf32, 1>, %arg4: memref<20x10xf32, 1>, %arg5: memref<f32, 1>, %arg6: memref<f32, 1>, %arg7: memref<f32, 1>, %arg8: memref<f32, 1>, %arg9: memref<20x10xf32, 1>, %arg10: memref<20x10xf32, 1>, %arg11: memref<20x10xf32, 1>, %arg12: memref<20x10xf32, 1>):
    %0 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<f32, 1>
    "affine.store"(%0, %1) {map = #map0} : (f32, memref<f32, 1>) -> ()
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%arg6) {map = #map0} : (memref<f32, 1>) -> f32
        "affine.store"(%22, %2, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%2, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%arg3, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.mulf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %3, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<f32, 1>
    %5 = "affine.load"(%arg6) {map = #map0} : (memref<f32, 1>) -> f32
    %6 = "affine.load"(%1) {map = #map0} : (memref<f32, 1>) -> f32
    %7 = "arith.subf"(%6, %5) : (f32, f32) -> f32
    "affine.store"(%7, %4) {map = #map0} : (f32, memref<f32, 1>) -> ()
    %8 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%4) {map = #map0} : (memref<f32, 1>) -> f32
        "affine.store"(%22, %8, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %9 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%arg1, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%8, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.mulf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %9, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %10 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%arg1, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%9, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.mulf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %10, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%10, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%3, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.addf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %arg11, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %11 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%2, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%arg2, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.mulf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %11, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%9, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%11, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.addf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %arg10, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %12 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%arg10, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%arg10, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.mulf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %12, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %13 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%12, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%arg11, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.subf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %13, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %14 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%arg7) {map = #map0} : (memref<f32, 1>) -> f32
        "affine.store"(%22, %14, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %15 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%arg4, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%14, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.mulf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %15, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %16 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%arg8) {map = #map0} : (memref<f32, 1>) -> f32
        "affine.store"(%22, %16, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %17 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%16, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%13, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.addf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %17, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %18 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%17, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "math.sqrt"(%22) : (f32) -> f32
        "affine.store"(%23, %18, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %19 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%arg5) {map = #map0} : (memref<f32, 1>) -> f32
        "affine.store"(%22, %19, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %20 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%arg1, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%19, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.mulf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %20, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    %21 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<20x10xf32, 1>
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%18, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%20, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.divf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %21, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%21, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%15, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.addf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %arg12, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "affine.for"() ({
    ^bb0(%arg13: index):
      "affine.for"() ({
      ^bb0(%arg14: index):
        %22 = "affine.load"(%arg12, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %23 = "affine.load"(%arg0, %arg13, %arg14) {map = #map1} : (memref<20x10xf32, 1>, index, index) -> f32
        %24 = "arith.subf"(%23, %22) : (f32, f32) -> f32
        "affine.store"(%24, %arg9, %arg13, %arg14) {map = #map1} : (f32, memref<20x10xf32, 1>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<20x10xf32, 1>, memref<20x10xf32, 1>, memref<20x10xf32, 1>, memref<20x10xf32, 1>, memref<20x10xf32, 1>, memref<f32, 1>, memref<f32, 1>, memref<f32, 1>, memref<f32, 1>, memref<20x10xf32, 1>, memref<20x10xf32, 1>, memref<20x10xf32, 1>, memref<20x10xf32, 1>) -> (), sym_name = "fuse_large_number_of_loops"} : () -> ()
}) : () -> ()

// -----
