"builtin.module"() ({
  "gpu.module"() ({
    "gpu.func"() ({
    ^bb0(%arg0: memref<6xi32>, %arg1: memref<6xi32>):
      %0 = "arith.constant"() {value = 2 : i32} : () -> i32
      %1 = "arith.constant"() {value = 0 : index} : () -> index
      %2 = "arith.constant"() {value = 1 : index} : () -> index
      %3 = "arith.constant"() {value = 2 : index} : () -> index
      %4 = "arith.constant"() {value = 3 : index} : () -> index
      %5 = "arith.constant"() {value = 4 : index} : () -> index
      %6 = "arith.constant"() {value = 5 : index} : () -> index
      %7 = "memref.load"(%arg0, %1) : (memref<6xi32>, index) -> i32
      %8 = "memref.load"(%arg0, %2) : (memref<6xi32>, index) -> i32
      %9 = "memref.load"(%arg0, %3) : (memref<6xi32>, index) -> i32
      %10 = "memref.load"(%arg0, %4) : (memref<6xi32>, index) -> i32
      %11 = "memref.load"(%arg0, %5) : (memref<6xi32>, index) -> i32
      %12 = "memref.load"(%arg0, %6) : (memref<6xi32>, index) -> i32
      %13 = "arith.muli"(%7, %0) : (i32, i32) -> i32
      %14 = "arith.muli"(%8, %0) : (i32, i32) -> i32
      %15 = "arith.muli"(%9, %0) : (i32, i32) -> i32
      %16 = "arith.muli"(%10, %0) : (i32, i32) -> i32
      %17 = "arith.muli"(%11, %0) : (i32, i32) -> i32
      %18 = "arith.muli"(%12, %0) : (i32, i32) -> i32
      "memref.store"(%13, %arg1, %1) : (i32, memref<6xi32>, index) -> ()
      "memref.store"(%14, %arg1, %2) : (i32, memref<6xi32>, index) -> ()
      "memref.store"(%15, %arg1, %3) : (i32, memref<6xi32>, index) -> ()
      "memref.store"(%16, %arg1, %4) : (i32, memref<6xi32>, index) -> ()
      "memref.store"(%17, %arg1, %5) : (i32, memref<6xi32>, index) -> ()
      "memref.store"(%18, %arg1, %6) : (i32, memref<6xi32>, index) -> ()
      "gpu.return"() : () -> ()
    }) {function_type = (memref<6xi32>, memref<6xi32>) -> (), gpu.kernel, spv.entry_point_abi = {local_size = dense<1> : vector<3xi32>}, sym_name = "double", workgroup_attributions = 0 : i64} : () -> ()
    "gpu.module_end"() : () -> ()
  }) {sym_name = "kernels"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<6xi32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<6xi32>
    %2 = "arith.constant"() {value = 4 : i32} : () -> i32
    %3 = "arith.constant"() {value = 0 : i32} : () -> i32
    %4 = "memref.cast"(%0) : (memref<6xi32>) -> memref<?xi32>
    %5 = "memref.cast"(%1) : (memref<6xi32>) -> memref<?xi32>
    "func.call"(%4, %2) {callee = @fillI32Buffer} : (memref<?xi32>, i32) -> ()
    "func.call"(%5, %3) {callee = @fillI32Buffer} : (memref<?xi32>, i32) -> ()
    %6 = "arith.constant"() {value = 1 : index} : () -> index
    "gpu.launch_func"(%6, %6, %6, %6, %6, %6, %0, %1) {kernel = @kernels::@double, operand_segment_sizes = dense<[0, 1, 1, 1, 1, 1, 1, 0, 2]> : vector<9xi32>} : (index, index, index, index, index, index, memref<6xi32>, memref<6xi32>) -> ()
    %7 = "memref.cast"(%1) : (memref<6xi32>) -> memref<*xi32>
    "func.call"(%7) {callee = @print_memref_i32} : (memref<*xi32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<?xi32>, i32) -> (), sym_name = "fillI32Buffer", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<*xi32>) -> (), sym_name = "print_memref_i32", sym_visibility = "private"} : () -> ()
}) {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_variable_pointers]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} : () -> ()

// -----
