"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.erase_this_arg}], function_type = (f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.erase_this_arg}, {test.A}], function_type = (f32, f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}, {test.erase_this_arg}], function_type = (f32, f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}, {test.erase_this_arg}, {test.B}], function_type = (f32, f32, f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}, {test.erase_this_arg}, {test.erase_this_arg}, {test.B}], function_type = (f32, f32, f32, f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}, {test.erase_this_arg}, {test.B}, {test.erase_this_arg}, {test.C}], function_type = (f32, f32, f32, f32, f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<1xf32>, %arg1: f32, %arg2: tensor<2xf32>, %arg3: f32, %arg4: tensor<3xf32>):
    "func.return"() : () -> ()
  }) {arg_attrs = [{}, {test.erase_this_arg}, {}, {test.erase_this_arg}, {}], function_type = (tensor<1xf32>, f32, tensor<2xf32>, f32, tensor<3xf32>) -> (), sym_name = "f"} : () -> ()
}) : () -> ()

// -----
