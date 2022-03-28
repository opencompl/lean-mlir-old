"builtin.module"() ({
  "func.func"() ({
    %0 = "foo"() : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "function"} : () -> ()
}) : () -> ()


