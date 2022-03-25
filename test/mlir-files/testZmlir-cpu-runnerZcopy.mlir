#map0 = affine_map<(d0, d1) -> (d0 + d1 * 2)>
#map1 = affine_map<(d0, d1, d2) -> (d0 * 3 + d1 + d2)>
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (memref<*xf32>) -> (), llvm.emit_c_interface, sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 4.200000e+01 : f32} : () -> f32
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<2x3xf32>
    %4 = "memref.dim"(%3, %0) : (memref<2x3xf32>, index) -> index
    %5 = "memref.dim"(%3, %1) : (memref<2x3xf32>, index) -> index
    "scf.parallel"(%0, %0, %4, %5, %1, %1) ({
    ^bb0(%arg0: index, %arg1: index):
      %19 = "arith.muli"(%arg0, %5) : (index, index) -> index
      %20 = "arith.addi"(%19, %arg1) : (index, index) -> index
      %21 = "arith.index_cast"(%20) : (index) -> i64
      %22 = "arith.sitofp"(%21) : (i64) -> f32
      "memref.store"(%22, %3, %arg0, %arg1) : (f32, memref<2x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) {operand_segment_sizes = dense<[2, 2, 2, 0]> : vector<4xi32>} : (index, index, index, index, index, index) -> ()
    %6 = "memref.cast"(%3) : (memref<2x3xf32>) -> memref<*xf32>
    "func.call"(%6) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    %7 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<2x3xf32>
    "memref.copy"(%3, %7) : (memref<2x3xf32>, memref<2x3xf32>) -> ()
    %8 = "memref.cast"(%7) : (memref<2x3xf32>) -> memref<*xf32>
    "func.call"(%8) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    %9 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<3x2xf32>
    %10 = "memref.reinterpret_cast"(%9) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, static_offsets = [0], static_sizes = [2, 3], static_strides = [1, 2]} : (memref<3x2xf32>) -> memref<2x3xf32, #map0>
    "memref.copy"(%3, %10) : (memref<2x3xf32>, memref<2x3xf32, #map0>) -> ()
    %11 = "memref.cast"(%9) : (memref<3x2xf32>) -> memref<*xf32>
    "func.call"(%11) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    %12 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<3x0x1xf32>
    %13 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<3x0x1xf32>
    "memref.copy"(%12, %13) : (memref<3x0x1xf32>, memref<3x0x1xf32>) -> ()
    %14 = "memref.reinterpret_cast"(%12) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, static_offsets = [0], static_sizes = [0, 3, 1], static_strides = [3, 1, 1]} : (memref<3x0x1xf32>) -> memref<0x3x1xf32, #map1>
    %15 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<0x3x1xf32>
    "memref.copy"(%14, %15) : (memref<0x3x1xf32, #map1>, memref<0x3x1xf32>) -> ()
    %16 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<f32>
    "memref.store"(%2, %16) : (f32, memref<f32>) -> ()
    %17 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<f32>
    "memref.copy"(%16, %17) : (memref<f32>, memref<f32>) -> ()
    %18 = "memref.cast"(%17) : (memref<f32>) -> memref<*xf32>
    "func.call"(%18) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    "memref.dealloc"(%13) : (memref<3x0x1xf32>) -> ()
    "memref.dealloc"(%15) : (memref<0x3x1xf32>) -> ()
    "memref.dealloc"(%12) : (memref<3x0x1xf32>) -> ()
    "memref.dealloc"(%9) : (memref<3x2xf32>) -> ()
    "memref.dealloc"(%7) : (memref<2x3xf32>) -> ()
    "memref.dealloc"(%3) : (memref<2x3xf32>) -> ()
    "memref.dealloc"(%16) : (memref<f32>) -> ()
    "memref.dealloc"(%17) : (memref<f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()

// -----
