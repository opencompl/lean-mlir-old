"builtin.module"() ({
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "f"} : () -> ()
  "builtin.module"() ({
    "func.func"() ({
      "func.return"() : () -> ()
    }) {function_type = () -> (), sym_name = "foo"} : () -> ()
    "func.func"() ({
      "func.return"() : () -> ()
    }) {function_type = () -> (), sym_name = "baz"} : () -> ()
  }) {sym_name = "inner_mod1"} : () -> ()
  "builtin.module"() ({
    "func.func"() ({
      "func.return"() : () -> ()
    }) {function_type = () -> (), sym_name = "foo"} : () -> ()
  }) {sym_name = "inner_mod2"} : () -> ()
}) : () -> ()


