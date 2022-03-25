"builtin.module"() ({
  "gpu.module"() ({
    "gpu.func"() ({
    ^bb0(%arg0: memref<8 × i32>, %arg1: memref<8x8xi32>, %arg2: memref<8x8x8xi32>):
      %0 = "gpu.block_id"() {dimension = #gpu<"dim  × ">} : () -> index
      %1 = "gpu.block_id"() {dimension = #gpu<"dim y">} : () -> index
      %2 = "gpu.block_id"() {dimension = #gpu<"dim z">} : () -> index
      %3 = "memref.load"(%arg0, %0) : (memref<8 × i32>, index) -> i32
      %4 = "memref.load"(%arg1, %1, %0) : (memref<8x8xi32>, index, index) -> i32
      %5 = "arith.addi"(%3, %4) : (i32, i32) -> i32
      "memref.store"(%5, %arg2, %2, %1, %0) : (i32, memref<8x8x8xi32>, index, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {function_type = (memref<8 × i32>, memref<8x8xi32>, memref<8x8x8xi32>) -> (), gpu.kernel, spv.entry_point_abi = {local_size = dense<1> : vector<3 × i32>}, sym_name = "kernel_addi", workgroup_attributions = 0 : i64} : () -> ()
    "gpu.module_end"() : () -> ()
  }) {sym_name = "kernels"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<8 × i32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<8x8xi32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<8x8x8xi32>
    %3 = "arith.constant"() {value = 0 : i32} : () -> i32
    %4 = "arith.constant"() {value = 1 : i32} : () -> i32
    %5 = "arith.constant"() {value = 2 : i32} : () -> i32
    %6 = "memref.cast"(%0) : (memref<8 × i32>) -> memref<? × i32>
    %7 = "memref.cast"(%1) : (memref<8x8xi32>) -> memref<?x?xi32>
    %8 = "memref.cast"(%2) : (memref<8x8x8xi32>) -> memref<?x?x?xi32>
    "func.call"(%6, %4) {callee = @fillResource1DInt} : (memref<? × i32>, i32) -> ()
    "func.call"(%7, %5) {callee = @fillResource2DInt} : (memref<?x?xi32>, i32) -> ()
    "func.call"(%8, %3) {callee = @fillResource3DInt} : (memref<?x?x?xi32>, i32) -> ()
    %9 = "arith.constant"() {value = 1 : index} : () -> index
    %10 = "arith.constant"() {value = 8 : index} : () -> index
    "gpu.launch_func"(%10, %10, %10, %9, %9, %9, %0, %1, %2) {kernel = @kernels::@kernel_addi, operand_segment_sizes = dense<[0, 1, 1, 1, 1, 1, 1, 0, 3]> : vector<9 × i32>} : (index, index, index, index, index, index, memref<8 × i32>, memref<8x8xi32>, memref<8x8x8xi32>) -> ()
    %11 = "memref.cast"(%8) : (memref<?x?x?xi32>) -> memref<* × i32>
    "func.call"(%11) {callee = @print_memref_i32} : (memref<* × i32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<? × i32>, i32) -> (), sym_name = "fillResource1DInt", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<?x?xi32>, i32) -> (), sym_name = "fillResource2DInt", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<?x?x?xi32>, i32) -> (), sym_name = "fillResource3DInt", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<* × i32>) -> (), sym_name = "print_memref_i32", sym_visibility = "private"} : () -> ()
}) {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>} : () -> ()

// -----
