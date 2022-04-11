"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (memref<* × f32>) -> (), llvm.emit_c_interface, sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<* × i32>) -> (), llvm.emit_c_interface, sym_name = "print_memref_i32", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "printNewline", sym_visibility = "private"} : () -> ()
  "memref.global"() {initial_value = dense<[0.00000000, 1.00000000, 2.00000000, 3.00000000]> : tensor<4 × f32>, sym_name = "gv0", sym_visibility = "private", type = memref<4 × f32>} : () -> ()
  "func.func"() ({
    %0 = "memref.get_global"() {name = @gv0} : () -> memref<4 × f32>
    %1 = "memref.cast"(%0) : (memref<4 × f32>) -> memref<* × f32>
    "func.call"(%1) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.call"() {callee = @printNewline} : () -> ()
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    %3 = "arith.constant"() {value = 2 : index} : () -> index
    %4 = "arith.constant"() {value = 4.00000000 : f32} : () -> f32
    %5 = "arith.constant"() {value = 5.00000000 : f32} : () -> f32
    "memref.store"(%4, %0, %2) : (f32, memref<4 × f32>, index) -> ()
    "memref.store"(%5, %0, %3) : (f32, memref<4 × f32>, index) -> ()
    "func.call"(%1) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.call"() {callee = @printNewline} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test1DMemref"} : () -> ()
  "memref.global"() {constant, initial_value = dense<[[0, 1], [2, 3], [4, 5]]> : tensor<3 × 2 × i32>, sym_name = "gv1", type = memref<3 × 2 × i32>} : () -> ()
  "func.func"() ({
    %0 = "memref.get_global"() {name = @gv1} : () -> memref<3 × 2 × i32>
    %1 = "memref.cast"(%0) : (memref<3 × 2 × i32>) -> memref<* × i32>
    "func.call"(%1) {callee = @print_memref_i32} : (memref<* × i32>) -> ()
    "func.call"() {callee = @printNewline} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "testConstantMemref"} : () -> ()
  "memref.global"() {initial_value = dense<[[0.00000000, 1.00000000], [2.00000000, 3.00000000], [4.00000000, 5.00000000], [6.00000000, 7.00000000]]> : tensor<4 × 2 × f32>, sym_name = "gv2", sym_visibility = "private", type = memref<4 × 2 × f32>} : () -> ()
  "func.func"() ({
    %0 = "memref.get_global"() {name = @gv2} : () -> memref<4 × 2 × f32>
    %1 = "memref.cast"(%0) : (memref<4 × 2 × f32>) -> memref<* × f32>
    "func.call"(%1) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.call"() {callee = @printNewline} : () -> ()
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    %3 = "arith.constant"() {value = 1 : index} : () -> index
    %4 = "arith.constant"() {value = 1.00000001 : f32} : () -> f32
    "memref.store"(%4, %0, %2, %3) : (f32, memref<4 × 2 × f32>, index, index) -> ()
    "func.call"(%1) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.call"() {callee = @printNewline} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test2DMemref"} : () -> ()
  "memref.global"() {initial_value = dense<11> : tensor<i32>, sym_name = "gv3", type = memref<i32>} : () -> ()
  "func.func"() ({
    %0 = "memref.get_global"() {name = @gv3} : () -> memref<i32>
    %1 = "memref.cast"(%0) : (memref<i32>) -> memref<* × i32>
    "func.call"(%1) {callee = @print_memref_i32} : (memref<* × i32>) -> ()
    "func.call"() {callee = @printNewline} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "testScalarMemref"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @test1DMemref} : () -> ()
    "func.call"() {callee = @testConstantMemref} : () -> ()
    "func.call"() {callee = @test2DMemref} : () -> ()
    "func.call"() {callee = @testScalarMemref} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()


