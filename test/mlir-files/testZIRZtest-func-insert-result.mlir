"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "f", sym_visibility = "private", test.insert_results = [[0, f32, {test.A}]]} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> f32, res_attrs = [{test.B}], sym_name = "f", sym_visibility = "private", test.insert_results = [[0, f32, {test.A}]]} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> f32, res_attrs = [{test.A}], sym_name = "f", sym_visibility = "private", test.insert_results = [[1, f32, {test.B}]]} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (f32, f32), res_attrs = [{test.A}, {test.C}], sym_name = "f", sym_visibility = "private", test.insert_results = [[1, f32, {test.B}]]} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> f32, res_attrs = [{test.B}], sym_name = "f", sym_visibility = "private", test.insert_results = [[0, f32, {test.A}], [1, f32, {test.C}]]} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> f32, res_attrs = [{test.C}], sym_name = "f", sym_visibility = "private", test.insert_results = [[0, f32, {test.A}], [0, f32, {test.B}]]} : () -> ()
}) : () -> ()

// -----
