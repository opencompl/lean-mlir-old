"builtin.module"() ({
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "func_a"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "func_b", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @func_b} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "func_c"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @func_c} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "func_d"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @func_c} : () -> ()
    "func.call"() {callee = @func_d} : () -> ()
    "func.call"() {callee = @func_e} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "func_e"} : () -> ()
  "func.func"() ({
    %0 = "test.functional_region_op"() ({
      "func.call"() {callee = @func_f} : () -> ()
      "test.return"() : () -> ()
    }) : () -> (() -> ())
    "func.call_indirect"(%0) : (() -> ()) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "func_f"} : () -> ()
}) {test.name = "simple"} : () -> ()


"builtin.module"() ({
  "builtin.module"() ({
    "func.func"() ({
      "func.return"() : () -> ()
    }) {function_type = () -> (), sym_name = "func_a"} : () -> ()
  }) {sym_name = "nested_module"} : () -> ()
  "func.func"() ({
    "test.conversion_call_op"() {callee = @nested_module::@func_a} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "func_b"} : () -> ()
}) {test.name = "nested"} : () -> ()


