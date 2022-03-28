"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.addi"(%arg0, %arg0) : (i32, i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "func_with_arg"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "func.call"(%arg0) {callee = @func_with_arg} : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "inline_with_arg"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    "func.return"(%0) : (i32) -> ()
  ^bb2:  
    %1 = "arith.constant"() {value = 55 : i32} : () -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i1) -> i32, sym_name = "func_with_multi_return"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = false} : () -> i1
    %1 = "func.call"(%0) {callee = @func_with_multi_return} : (i1) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "inline_with_multi_return"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.addi"(%arg0, %arg0) : (i32, i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "func_with_locations"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "func.call"(%arg0) {callee = @func_with_locations} : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "inline_with_locations"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "func_external", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @func_external} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "no_inline_external"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "multilevel_func_a"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @multilevel_func_a} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "multilevel_func_b"} : () -> ()
  "func.func"() ({
    %0 = "test.functional_region_op"() ({
      "func.call"() {callee = @multilevel_func_b} : () -> ()
      "test.return"() : () -> ()
    }) : () -> (() -> ())
    "func.call_indirect"(%0) : (() -> ()) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "inline_multilevel"} : () -> ()
  "func.func"() ({
    %0 = "test.functional_region_op"() ({
      "func.call"() {callee = @no_inline_recursive} : () -> ()
      "test.return"() : () -> ()
    }) : () -> (() -> ())
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "no_inline_recursive"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    "func.return"(%arg0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "convert_callee_fn"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    "func.return"() : () -> ()
  }) {function_type = (i32, i32) -> (), sym_name = "convert_callee_fn_multi_arg"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    "func.return"(%0, %0) : (i32, i32) -> ()
  }) {function_type = () -> (i32, i32), sym_name = "convert_callee_fn_multi_res"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i16} : () -> i16
    %1 = "test.conversion_call_op"(%0) {callee = @convert_callee_fn} : (i16) -> i16
    "func.return"(%1) : (i16) -> ()
  }) {function_type = () -> i16, sym_name = "inline_convert_call"} : () -> ()
  "func.func"() ({
    "cf.br"()[^bb1] : () -> ()
  ^bb1:  
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "convert_callee_fn_multiblock"} : () -> ()
  "func.func"() ({
    %0 = "test.conversion_call_op"() {callee = @convert_callee_fn_multiblock} : () -> i16
    "func.return"(%0) : (i16) -> ()
  }) {function_type = () -> i16, sym_name = "inline_convert_result_multiblock"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i16} : () -> i16
    %1 = "arith.constant"() {value = 0 : i64} : () -> i64
    "test.conversion_call_op"(%0, %1) {callee = @convert_callee_fn_multi_arg} : (i16, i64) -> ()
    %2:2 = "test.conversion_call_op"() {callee = @convert_callee_fn_multi_res} : () -> (i16, i64)
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "no_inline_convert_call"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "simplify_return_constant"} : () -> ()
  "func.func"() ({
    %0 = "func.constant"() {value = @simplify_return_constant} : () -> (() -> i32)
    "func.return"(%0) : (() -> i32) -> ()
  }) {function_type = () -> (() -> i32), sym_name = "simplify_return_reference"} : () -> ()
  "func.func"() ({
    %0 = "func.call"() {callee = @simplify_return_reference} : () -> (() -> i32)
    %1 = "func.call_indirect"(%0) : (() -> i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "inline_simplify"} : () -> ()
  "func.func"() ({
    %0 = "test.conversion_call_op"() {callee = @convert_callee_fn_multiblock, noinline} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "no_inline_invalid_call"} : () -> ()
  "func.func"() ({
    %0 = "gpu.alloc"() {operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> memref<1024 × f32>
    "func.return"(%0) : (memref<1024 × f32>) -> ()
  }) {function_type = () -> memref<1024 × f32>, sym_name = "gpu_alloc"} : () -> ()
  "func.func"() ({
    %0 = "func.call"() {callee = @gpu_alloc} : () -> memref<1024 × f32>
    "func.return"(%0) : (memref<1024 × f32>) -> ()
  }) {function_type = () -> memref<1024 × f32>, sym_name = "inline_gpu_ops"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    "cf.br"(%arg0)[^bb1] : (i32) -> ()
  ^bb1(%0: i32):  
    "test.foo"(%0) : (i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32) -> (), sym_name = "func_with_block_args_location"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    "func.call"(%arg0) {callee = @func_with_block_args_location} : (i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32) -> (), sym_name = "func_with_block_args_location_callee1"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    "func.call"(%arg0) {callee = @func_with_block_args_location} : (i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32) -> (), sym_name = "func_with_block_args_location_callee2"} : () -> ()
}) : () -> ()


