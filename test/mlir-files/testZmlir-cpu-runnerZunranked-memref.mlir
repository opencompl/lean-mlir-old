"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<10x3xf32>
    %1 = "arith.constant"() {value = 2.000000e+00 : f32} : () -> f32
    %2 = "arith.constant"() {value = 5.000000e+00 : f32} : () -> f32
    %3 = "arith.constant"() {value = 1.000000e+01 : f32} : () -> f32
    %4 = "memref.cast"(%0) : (memref<10x3xf32>) -> memref<?x?xf32>
    "linalg.fill"(%3, %4) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2xi32>} : (f32, memref<?x?xf32>) -> ()
    %5 = "memref.cast"(%0) : (memref<10x3xf32>) -> memref<*xf32>
    "func.call"(%5) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    %6 = "memref.cast"(%5) : (memref<*xf32>) -> memref<?x?xf32>
    "linalg.fill"(%2, %6) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2xi32>} : (f32, memref<?x?xf32>) -> ()
    %7 = "memref.cast"(%6) : (memref<?x?xf32>) -> memref<*xf32>
    "func.call"(%7) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    %8 = "memref.cast"(%6) : (memref<?x?xf32>) -> memref<*xf32>
    %9 = "memref.cast"(%8) : (memref<*xf32>) -> memref<?x?xf32>
    "linalg.fill"(%1, %9) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2xi32>} : (f32, memref<?x?xf32>) -> ()
    %10 = "memref.cast"(%6) : (memref<?x?xf32>) -> memref<*xf32>
    "func.call"(%10) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    %11 = "arith.constant"() {value = 122 : i8} : () -> i8
    %12 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<i8>
    "memref.store"(%11, %12) : (i8, memref<i8>) -> ()
    %13 = "memref.cast"(%12) : (memref<i8>) -> memref<*xi8>
    "func.call"(%13) {callee = @print_memref_i8} : (memref<*xi8>) -> ()
    "memref.dealloc"(%13) : (memref<*xi8>) -> ()
    "memref.dealloc"(%0) : (memref<10x3xf32>) -> ()
    "func.call"() {callee = @return_var_memref_caller} : () -> ()
    "func.call"() {callee = @return_two_var_memref_caller} : () -> ()
    "func.call"() {callee = @dim_op_of_unranked} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<*xi8>) -> (), llvm.emit_c_interface, sym_name = "print_memref_i8", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<*xf32>) -> (), llvm.emit_c_interface, sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<4x3xf32>
    %1 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    "linalg.fill"(%1, %0) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2xi32>} : (f32, memref<4x3xf32>) -> ()
    %2:2 = "func.call"(%0) {callee = @return_two_var_memref} : (memref<4x3xf32>) -> (memref<*xf32>, memref<*xf32>)
    "func.call"(%2#0) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    "func.call"(%2#1) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "return_two_var_memref_caller"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<4x3xf32>):
    %0 = "memref.cast"(%arg0) : (memref<4x3xf32>) -> memref<*xf32>
    "func.return"(%0, %0) : (memref<*xf32>, memref<*xf32>) -> ()
  }) {function_type = (memref<4x3xf32>) -> (memref<*xf32>, memref<*xf32>), sym_name = "return_two_var_memref"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<4x3xf32>
    %1 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    "linalg.fill"(%1, %0) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2xi32>} : (f32, memref<4x3xf32>) -> ()
    %2 = "func.call"(%0) {callee = @return_var_memref} : (memref<4x3xf32>) -> memref<*xf32>
    "func.call"(%2) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "return_var_memref_caller"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<4x3xf32>):
    %0 = "memref.cast"(%arg0) : (memref<4x3xf32>) -> memref<*xf32>
    "func.return"(%0) : (memref<*xf32>) -> ()
  }) {function_type = (memref<4x3xf32>) -> memref<*xf32>, sym_name = "return_var_memref"} : () -> ()
  "func.func"() ({
  }) {function_type = (index) -> (), sym_name = "printU64", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "printNewline", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<4x3xf32>
    %1 = "memref.cast"(%0) : (memref<4x3xf32>) -> memref<*xf32>
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    %3 = "memref.dim"(%1, %2) : (memref<*xf32>, index) -> index
    "func.call"(%3) {callee = @printU64} : (index) -> ()
    "func.call"() {callee = @printNewline} : () -> ()
    %4 = "arith.constant"() {value = 1 : index} : () -> index
    %5 = "memref.dim"(%1, %4) : (memref<*xf32>, index) -> index
    "func.call"(%5) {callee = @printU64} : (index) -> ()
    "func.call"() {callee = @printNewline} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dim_op_of_unranked"} : () -> ()
}) : () -> ()

// -----
