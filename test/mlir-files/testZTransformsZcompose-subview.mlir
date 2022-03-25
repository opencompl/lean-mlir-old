#map0 = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 2304)>
#map1 = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 3456)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<4x1024xf32>):
    %0 = "memref.subview"(%arg0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [2, 256], static_sizes = [2, 256], static_strides = [1, 1]} : (memref<4x1024xf32>) -> memref<2x256xf32, #map0>
    %1 = "memref.subview"(%0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [1, 128], static_sizes = [1, 128], static_strides = [1, 1]} : (memref<2x256xf32, #map0>) -> memref<1x128xf32, #map1>
    "func.return"(%1) : (memref<1x128xf32, #map1>) -> ()
  }) {function_type = (memref<4x1024xf32>) -> memref<1x128xf32, #map1>, sym_name = "main"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 1536)>
#map1 = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 2688)>
#map2 = affine_map<(d0, d1) -> (d0 * 1024 + d1 + 3745)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<4x1024xf32>):
    %0 = "memref.subview"(%arg0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [1, 512], static_sizes = [3, 256], static_strides = [1, 1]} : (memref<4x1024xf32>) -> memref<3x256xf32, #map0>
    %1 = "memref.subview"(%0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [1, 128], static_sizes = [2, 128], static_strides = [1, 1]} : (memref<3x256xf32, #map0>) -> memref<2x128xf32, #map1>
    %2 = "memref.subview"(%1) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [1, 33], static_sizes = [1, 10], static_strides = [1, 1]} : (memref<2x128xf32, #map1>) -> memref<1x10xf32, #map2>
    "func.return"(%2) : (memref<1x10xf32, #map2>) -> ()
  }) {function_type = (memref<4x1024xf32>) -> memref<1x10xf32, #map2>, sym_name = "main"} : () -> ()
}) : () -> ()

// -----
#map = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<4x1024xf32>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 2 : index} : () -> index
    %2 = "memref.subview"(%arg0, %1) {operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4 × i32>, static_offsets = [-9223372036854775808, 256], static_sizes = [2, 256], static_strides = [1, 1]} : (memref<4x1024xf32>, index) -> memref<2x256xf32, #map>
    %3 = "memref.subview"(%2, %0) {operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4 × i32>, static_offsets = [-9223372036854775808, 128], static_sizes = [1, 128], static_strides = [1, 1]} : (memref<2x256xf32, #map>, index) -> memref<1x128xf32, #map>
    "func.return"(%3) : (memref<1x128xf32, #map>) -> ()
  }) {function_type = (memref<4x1024xf32>) -> memref<1x128xf32, #map>, sym_name = "main"} : () -> ()
}) : () -> ()

// -----
#map = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<4x1024xf32>):
    %0 = "arith.constant"() {value = 2 : index} : () -> index
    %1 = "arith.constant"() {value = 128 : index} : () -> index
    %2 = "memref.subview"(%arg0, %0) {operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4 × i32>, static_offsets = [-9223372036854775808, 256], static_sizes = [2, 256], static_strides = [1, 1]} : (memref<4x1024xf32>, index) -> memref<2x256xf32, #map>
    %3 = "memref.subview"(%2, %1) {operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4 × i32>, static_offsets = [1, -9223372036854775808], static_sizes = [1, 128], static_strides = [1, 1]} : (memref<2x256xf32, #map>, index) -> memref<1x128xf32, #map>
    "func.return"(%3) : (memref<1x128xf32, #map>) -> ()
  }) {function_type = (memref<4x1024xf32>) -> memref<1x128xf32, #map>, sym_name = "main"} : () -> ()
}) : () -> ()

// -----
