"builtin.module"() ({
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "f", test.insert_args = [[0, i1, {test.A}]]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i2):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.B}], function_type = (i2) -> (), sym_name = "f", test.insert_args = [[0, i1, {test.A}]]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}], function_type = (i1) -> (), sym_name = "f", test.insert_args = [[1, i2, {test.B}]]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i3):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}, {test.C}], function_type = (i1, i3) -> (), sym_name = "f", test.insert_args = [[1, i2, {test.B}]]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i2):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.B}], function_type = (i2) -> (), sym_name = "f", test.insert_args = [[0, i1, {test.A}], [1, i3, {test.C}]]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i3):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.C}], function_type = (i3) -> (), sym_name = "f", test.insert_args = [[0, i1, {test.A}], [0, i2, {test.B}]]} : () -> ()
}) : () -> ()


