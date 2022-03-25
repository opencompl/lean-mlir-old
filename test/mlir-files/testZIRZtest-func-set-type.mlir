"builtin.module"() ({
^bb0:
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (f32) -> (), sym_name = "t", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {arg_attrs = [{test.A}, {test.B}], function_type = (f32, f32) -> (), sym_name = "erase_arg", sym_visibility = "private", test.set_type_from = @t} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> f32, sym_name = "t", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (f32, f32), res_attrs = [{test.A}, {test.B}], sym_name = "erase_result", sym_visibility = "private", test.set_type_from = @t} : () -> ()
}) : () -> ()

// -----
