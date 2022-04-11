





"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1x16x14x14xf32, #map0>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1x16x14x14xf32, #map0>
    "test.op_norm"(%arg0, %0) : (memref<1x16x14x14xf32, #map0>, memref<1x16x14x14xf32, #map0>) -> ()
    "memref.dealloc"(%0) : (memref<1x16x14x14xf32, #map0>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1x16x14x14xf32, #map0>) -> (), sym_name = "test_norm"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<1x16x14x14xf32, #map0>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1x16x14x14xf32, #map0>
    "test.op_nonnorm"(%arg0, %0) : (memref<1x16x14x14xf32, #map0>, memref<1x16x14x14xf32, #map0>) -> ()
    "memref.dealloc"(%0) : (memref<1x16x14x14xf32, #map0>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1x16x14x14xf32, #map0>) -> (), sym_name = "test_nonnorm"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<1x16x1x1x32x64xf32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1x16x14x14xf32, #map0>
    "test.op_norm"(%arg0, %0) : (memref<1x16x1x1x32x64xf32>, memref<1x16x14x14xf32, #map0>) -> ()
    "memref.dealloc"(%0) : (memref<1x16x14x14xf32, #map0>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1x16x1x1x32x64xf32>) -> (), sym_name = "test_norm_mix"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<1x16x14x14xf32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1x16x14x14xf32, #map1>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1x16x14x14xf32>
    "test.op_norm"(%0, %1) : (memref<1x16x14x14xf32, #map1>, memref<1x16x14x14xf32>) -> ()
    %2 = "arith.constant"() {value = 3.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        "affine.for"() ({
        ^bb0(%arg3: index):
          "affine.for"() ({
          ^bb0(%arg4: index):
            %3 = "memref.load"(%1, %arg1, %arg2, %arg3, %arg4) : (memref<1x16x14x14xf32>, index, index, index, index) -> f32
            %4 = "arith.addf"(%3, %2) : (f32, f32) -> f32
            "memref.store"(%4, %arg0, %arg1, %arg2, %arg3, %arg4) : (f32, memref<1x16x14x14xf32>, index, index, index, index) -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : () -> ()
    "memref.dealloc"(%0) : (memref<1x16x14x14xf32, #map1>) -> ()
    "memref.dealloc"(%1) : (memref<1x16x14x14xf32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1x16x14x14xf32>) -> (), sym_name = "test_load_store"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<1x16x14x14xf32, #map1>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1x16x14x14xf32, #map1>
    %1:2 = "test.op_norm_ret"(%arg0) : (memref<1x16x14x14xf32, #map1>) -> (memref<1x16x14x14xf32, #map1>, memref<1x16x14x14xf32>)
    "test.op_norm"(%1#0, %0) : (memref<1x16x14x14xf32, #map1>, memref<1x16x14x14xf32, #map1>) -> ()
    "memref.dealloc"(%0) : (memref<1x16x14x14xf32, #map1>) -> ()
    "func.return"(%1#0, %1#1) : (memref<1x16x14x14xf32, #map1>, memref<1x16x14x14xf32>) -> ()
  }) {function_type = (memref<1x16x14x14xf32, #map1>) -> (memref<1x16x14x14xf32, #map1>, memref<1x16x14x14xf32>), sym_name = "test_norm_ret"} : () -> ()
  "test.op_funcref"() {func = @test_norm_mix} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<3 × f32, #map>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<3 × f32>
    "test.op_norm"(%arg0, %0) : (memref<3 × f32, #map>, memref<3 × f32>) -> ()
    %1 = "memref.reinterpret_cast"(%0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [0], static_sizes = [3, 1, 1], static_strides = [1, 1, 1]} : (memref<3 × f32>) -> memref<3 × 1 × 1 × f32>
    "func.return"(%1) : (memref<3 × 1 × 1 × f32>) -> ()
  }) {function_type = (memref<3 × f32, #map>) -> memref<3 × 1 × 1 × f32>, sym_name = "test_norm_reinterpret_cast"} : () -> ()
}) : () -> ()


