"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 2 : index} : () -> index
    %3 = "arith.constant"() {value = 3 : index} : () -> index
    %4 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %5 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %6 = "arith.constant"() {value = 2.000000e+00 : f32} : () -> f32
    %7 = "arith.constant"() {value = 3.000000e+00 : f32} : () -> f32
    %8 = "arith.constant"() {value = 4.000000e+00 : f32} : () -> f32
    %9 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<4xf32>
    "linalg.fill"(%4, %9) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2xi32>} : (f32, memref<4xf32>) -> ()
    %10 = "memref.cast"(%9) : (memref<4xf32>) -> memref<*xf32>
    "func.call"(%10) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    "memref.store"(%5, %9, %0) : (f32, memref<4xf32>, index) -> ()
    "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
    "func.call"(%10) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    %11 = "async.execute"() ({
      "memref.store"(%6, %9, %1) : (f32, memref<4xf32>, index) -> ()
      "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
      "func.call"(%10) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
      %12 = "async.execute"() ({
        "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
        "async.yield"() : () -> ()
      }) {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> !async.token
      %13 = "async.execute"(%12) ({
        "memref.store"(%7, %9, %2) : (f32, memref<4xf32>, index) -> ()
        "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
        "func.call"(%10) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
        "async.yield"() : () -> ()
      }) {operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!async.token) -> !async.token
      "async.await"(%13) : (!async.token) -> ()
      "memref.store"(%8, %9, %3) : (f32, memref<4xf32>, index) -> ()
      "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
      "func.call"(%10) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
      "async.yield"() : () -> ()
    }) {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> !async.token
    "async.await"(%11) : (!async.token) -> ()
    "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
    "func.call"(%10) {callee = @print_memref_f32} : (memref<*xf32>) -> ()
    "memref.dealloc"(%9) : (memref<4xf32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "mlirAsyncRuntimePrintCurrentThreadId", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<*xf32>) -> (), llvm.emit_c_interface, sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
}) : () -> ()

// -----
