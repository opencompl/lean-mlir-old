#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 32, d3 floordiv 64, d2 mod 32, d3 mod 64)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1x?x?x14xf32, #map>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 2 : index} : () -> index
    %2 = "memref.dim"(%arg0, %0) : (memref<1x?x?x14xf32, #map>, index) -> index
    %3 = "memref.dim"(%arg0, %1) : (memref<1x?x?x14xf32, #map>, index) -> index
    %4 = "memref.alloc"(%2, %3) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<1x?x?x14xf32, #map>
    "test.op_norm"(%arg0, %4) : (memref<1x?x?x14xf32, #map>, memref<1x?x?x14xf32, #map>) -> ()
    "memref.dealloc"(%4) : (memref<1x?x?x14xf32, #map>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1x?x?x14xf32, #map>) -> (), sym_name = "test_norm_dynamic12"} : () -> ()
}) : () -> ()


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, (d2 floordiv 4) floordiv 32, (d3 mod 8) floordiv 64, (d2 floordiv 4) mod 32, (d3 mod 8) mod 64)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<?x?x?x?xf32, #map>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 2 : index} : () -> index
    %3 = "arith.constant"() {value = 3 : index} : () -> index
    %4 = "memref.dim"(%arg0, %0) : (memref<?x?x?x?xf32, #map>, index) -> index
    %5 = "memref.dim"(%arg0, %1) : (memref<?x?x?x?xf32, #map>, index) -> index
    %6 = "memref.dim"(%arg0, %2) : (memref<?x?x?x?xf32, #map>, index) -> index
    %7 = "memref.dim"(%arg0, %3) : (memref<?x?x?x?xf32, #map>, index) -> index
    %8 = "memref.alloc"(%4, %5, %6, %7) {operand_segment_sizes = dense<[4, 0]> : vector<2 × i32>} : (index, index, index, index) -> memref<?x?x?x?xf32, #map>
    "test.op_norm"(%arg0, %8) : (memref<?x?x?x?xf32, #map>, memref<?x?x?x?xf32, #map>) -> ()
    "memref.dealloc"(%8) : (memref<?x?x?x?xf32, #map>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<?x?x?x?xf32, #map>) -> (), sym_name = "test_norm_dynamic1234"} : () -> ()
}) : () -> ()


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - d1, d3 - d2)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1x?x?x14xf32, #map>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 2 : index} : () -> index
    %2 = "memref.dim"(%arg0, %0) : (memref<1x?x?x14xf32, #map>, index) -> index
    %3 = "memref.dim"(%arg0, %1) : (memref<1x?x?x14xf32, #map>, index) -> index
    %4 = "memref.alloc"(%2, %3) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<1x?x?x14xf32, #map>
    "test.op_norm"(%arg0, %4) : (memref<1x?x?x14xf32, #map>, memref<1x?x?x14xf32, #map>) -> ()
    "memref.dealloc"(%4) : (memref<1x?x?x14xf32, #map>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1x?x?x14xf32, #map>) -> (), sym_name = "test_norm_dynamic_not_tiled0"} : () -> ()
}) : () -> ()


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - d1, d3 - d2, d2 mod 32, d3 mod 64)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1x?x?x14xf32, #map>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 2 : index} : () -> index
    %2 = "memref.dim"(%arg0, %0) : (memref<1x?x?x14xf32, #map>, index) -> index
    %3 = "memref.dim"(%arg0, %1) : (memref<1x?x?x14xf32, #map>, index) -> index
    %4 = "memref.alloc"(%2, %3) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<1x?x?x14xf32, #map>
    "test.op_norm"(%arg0, %4) : (memref<1x?x?x14xf32, #map>, memref<1x?x?x14xf32, #map>) -> ()
    "memref.dealloc"(%4) : (memref<1x?x?x14xf32, #map>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1x?x?x14xf32, #map>) -> (), sym_name = "test_norm_dynamic_not_tiled1"} : () -> ()
}) : () -> ()


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - d1, d3 floordiv 64, d2 mod 32, d3 mod 32)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1x?x?x14xf32, #map>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 2 : index} : () -> index
    %2 = "memref.dim"(%arg0, %0) : (memref<1x?x?x14xf32, #map>, index) -> index
    %3 = "memref.dim"(%arg0, %1) : (memref<1x?x?x14xf32, #map>, index) -> index
    %4 = "memref.alloc"(%2, %3) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<1x?x?x14xf32, #map>
    "test.op_norm"(%arg0, %4) : (memref<1x?x?x14xf32, #map>, memref<1x?x?x14xf32, #map>) -> ()
    "memref.dealloc"(%4) : (memref<1x?x?x14xf32, #map>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1x?x?x14xf32, #map>) -> (), sym_name = "test_norm_dynamic_not_tiled2"} : () -> ()
}) : () -> ()


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 32, d2, d3, d1 mod 32, d1 mod 32)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1x?x?x14xf32, #map>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 2 : index} : () -> index
    %2 = "memref.dim"(%arg0, %0) : (memref<1x?x?x14xf32, #map>, index) -> index
    %3 = "memref.dim"(%arg0, %1) : (memref<1x?x?x14xf32, #map>, index) -> index
    %4 = "memref.alloc"(%2, %3) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<1x?x?x14xf32, #map>
    "test.op_norm"(%arg0, %4) : (memref<1x?x?x14xf32, #map>, memref<1x?x?x14xf32, #map>) -> ()
    "memref.dealloc"(%4) : (memref<1x?x?x14xf32, #map>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1x?x?x14xf32, #map>) -> (), sym_name = "test_norm_dynamic_not_tiled3"} : () -> ()
}) : () -> ()


#map = affine_map<(d0, d1, d2, d3) -> (d0 floordiv 32, d1 floordiv 32, d0, d3, d0 mod 32, d1 mod 32)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1x?x?x14xf32, #map>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 2 : index} : () -> index
    %2 = "memref.dim"(%arg0, %0) : (memref<1x?x?x14xf32, #map>, index) -> index
    %3 = "memref.dim"(%arg0, %1) : (memref<1x?x?x14xf32, #map>, index) -> index
    %4 = "memref.alloc"(%2, %3) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<1x?x?x14xf32, #map>
    "test.op_norm"(%arg0, %4) : (memref<1x?x?x14xf32, #map>, memref<1x?x?x14xf32, #map>) -> ()
    "memref.dealloc"(%4) : (memref<1x?x?x14xf32, #map>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1x?x?x14xf32, #map>) -> (), sym_name = "test_norm_dynamic_not_tiled4"} : () -> ()
}) : () -> ()


