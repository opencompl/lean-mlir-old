"builtin.module"() ({
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "f"} : () -> ()
  "builtin.module"() ({
    "builtin.module"() ({
    ^bb0:
    }) {sym_name = "foo"} : () -> ()
    "builtin.module"() ({
    ^bb0:
    }) {sym_name = "baz"} : () -> ()
  }) {sym_name = "inner_mod1"} : () -> ()
}) : () -> ()


