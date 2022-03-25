"builtin.module"() ({
  "gpu.module"() ({
    "gpu.func"() ({
    ^bb0(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>):
      %0 = "gpu.block_id"() {dimension = #gpu<"dim  × ">} : () -> index
      %1 = "gpu.block_id"() {dimension = #gpu<"dim y">} : () -> index
      %2 = "memref.load"(%arg0, %0, %1) : (memref<4x4xf32>, index, index) -> f32
      %3 = "memref.load"(%arg1, %0, %1) : (memref<4x4xf32>, index, index) -> f32
      %4 = "arith.mulf"(%2, %3) : (f32, f32) -> f32
      "memref.store"(%4, %arg2, %0, %1) : (f32, memref<4x4xf32>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {function_type = (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> (), gpu.kernel, spv.entry_point_abi = {local_size = dense<1> : vector<3 × i32>}, sym_name = "kernel_mul", workgroup_attributions = 0 : i64} : () -> ()
    "gpu.module_end"() : () -> ()
  }) {sym_name = "kernels"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<4x4xf32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<4x4xf32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<4x4xf32>
    %3 = "arith.constant"() {value = 0 : i32} : () -> i32
    %4 = "arith.constant"() {value = 1 : i32} : () -> i32
    %5 = "arith.constant"() {value = 2 : i32} : () -> i32
    %6 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %7 = "arith.constant"() {value = 2.000000e+00 : f32} : () -> f32
    %8 = "arith.constant"() {value = 3.000000e+00 : f32} : () -> f32
    %9 = "memref.cast"(%0) : (memref<4x4xf32>) -> memref<?x?xf32>
    %10 = "memref.cast"(%1) : (memref<4x4xf32>) -> memref<?x?xf32>
    %11 = "memref.cast"(%2) : (memref<4x4xf32>) -> memref<?x?xf32>
    "func.call"(%9, %7) {callee = @fillResource2DFloat} : (memref<?x?xf32>, f32) -> ()
    "func.call"(%10, %8) {callee = @fillResource2DFloat} : (memref<?x?xf32>, f32) -> ()
    "func.call"(%11, %6) {callee = @fillResource2DFloat} : (memref<?x?xf32>, f32) -> ()
    %12 = "arith.constant"() {value = 1 : index} : () -> index
    %13 = "arith.constant"() {value = 4 : index} : () -> index
    "gpu.launch_func"(%13, %13, %12, %12, %12, %12, %0, %1, %2) {kernel = @kernels::@kernel_mul, operand_segment_sizes = dense<[0, 1, 1, 1, 1, 1, 1, 0, 3]> : vector<9 × i32>} : (index, index, index, index, index, index, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
    %14 = "memref.cast"(%11) : (memref<?x?xf32>) -> memref<* × f32>
    "func.call"(%14) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<?x?xf32>, f32) -> (), sym_name = "fillResource2DFloat", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<* × f32>) -> (), sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
}) {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>} : () -> ()

// -----
