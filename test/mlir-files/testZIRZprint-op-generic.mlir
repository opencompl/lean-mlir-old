"builtin.module"() ({
  "func.func"() ({
    %0 = "test.string_attr_pretty_name"() {names = ["x"]} : () -> i32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "pretty_names"} : () -> ()
}) : () -> ()

// -----
