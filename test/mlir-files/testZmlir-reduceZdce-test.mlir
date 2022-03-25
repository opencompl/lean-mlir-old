"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "dead_private_function", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "dead_nested_function", sym_visibility = "nested"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "test.op_crash"() : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "simple1"} : () -> ()
}) : () -> ()

// -----
