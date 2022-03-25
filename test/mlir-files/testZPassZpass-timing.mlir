"builtin.module"() ({
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "foo"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "bar"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "baz"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "foobar"} : () -> ()
  "builtin.module"() ({
    "func.func"() ({
      "func.return"() : () -> ()
    }) {function_type = () -> (), sym_name = "baz"} : () -> ()
    "func.func"() ({
      "func.return"() : () -> ()
    }) {function_type = () -> (), sym_name = "foobar"} : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
