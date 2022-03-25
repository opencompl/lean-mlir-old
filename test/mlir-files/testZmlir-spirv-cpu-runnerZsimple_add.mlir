"builtin.module"() ({
  "gpu.module"() ({
    "gpu.func"() ({
    ^bb0(%arg0: memref<3 × f32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3x3xf32>):
      %0 = "arith.constant"() {value = 0 : index} : () -> index
      %1 = "arith.constant"() {value = 1 : index} : () -> index
      %2 = "arith.constant"() {value = 2 : index} : () -> index
      %3 = "memref.load"(%arg0, %0) : (memref<3 × f32>, index) -> f32
      %4 = "memref.load"(%arg1, %0, %0) : (memref<3x3xf32>, index, index) -> f32
      %5 = "arith.addf"(%3, %4) : (f32, f32) -> f32
      "memref.store"(%5, %arg2, %0, %0, %0) : (f32, memref<3x3x3xf32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %0, %1, %0) : (f32, memref<3x3x3xf32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %0, %2, %0) : (f32, memref<3x3x3xf32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %1, %0, %1) : (f32, memref<3x3x3xf32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %1, %1, %1) : (f32, memref<3x3x3xf32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %1, %2, %1) : (f32, memref<3x3x3xf32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %2, %0, %2) : (f32, memref<3x3x3xf32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %2, %1, %2) : (f32, memref<3x3x3xf32>, index, index, index) -> ()
      "memref.store"(%5, %arg2, %2, %2, %2) : (f32, memref<3x3x3xf32>, index, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {function_type = (memref<3 × f32>, memref<3x3xf32>, memref<3x3x3xf32>) -> (), gpu.kernel, spv.entry_point_abi = {local_size = dense<1> : vector<3 × i32>}, sym_name = "sum", workgroup_attributions = 0 : i64} : () -> ()
    "gpu.module_end"() : () -> ()
  }) {sym_name = "kernels"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<3 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<3x3xf32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<3x3x3xf32>
    %3 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %4 = "arith.constant"() {value = 3.400000e+00 : f32} : () -> f32
    %5 = "arith.constant"() {value = 4.300000e+00 : f32} : () -> f32
    %6 = "memref.cast"(%0) : (memref<3 × f32>) -> memref<? × f32>
    %7 = "memref.cast"(%1) : (memref<3x3xf32>) -> memref<?x?xf32>
    %8 = "memref.cast"(%2) : (memref<3x3x3xf32>) -> memref<?x?x?xf32>
    "func.call"(%6, %4) {callee = @fillF32Buffer1D} : (memref<? × f32>, f32) -> ()
    "func.call"(%7, %5) {callee = @fillF32Buffer2D} : (memref<?x?xf32>, f32) -> ()
    "func.call"(%8, %3) {callee = @fillF32Buffer3D} : (memref<?x?x?xf32>, f32) -> ()
    %9 = "arith.constant"() {value = 1 : index} : () -> index
    "gpu.launch_func"(%9, %9, %9, %9, %9, %9, %0, %1, %2) {kernel = @kernels::@sum, operand_segment_sizes = dense<[0, 1, 1, 1, 1, 1, 1, 0, 3]> : vector<9 × i32>} : (index, index, index, index, index, index, memref<3 × f32>, memref<3x3xf32>, memref<3x3x3xf32>) -> ()
    %10 = "memref.cast"(%2) : (memref<3x3x3xf32>) -> memref<* × f32>
    "func.call"(%10) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<? × f32>, f32) -> (), sym_name = "fillF32Buffer1D", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<?x?xf32>, f32) -> (), sym_name = "fillF32Buffer2D", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<?x?x?xf32>, f32) -> (), sym_name = "fillF32Buffer3D", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<* × f32>) -> (), sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
}) {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_8bit_storage]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3 × i32>}>} : () -> ()

// -----
