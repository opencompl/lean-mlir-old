"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (!test.test_type) -> i32, sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "test.type_producer"() : () -> !test.test_type
    %1 = "func.call"(%0) {callee = @callee} : (!test.test_type) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "caller"} : () -> ()
}) : () -> ()


