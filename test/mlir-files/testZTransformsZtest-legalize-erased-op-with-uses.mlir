"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.illegal_op_a"() : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "remove_all_ops"} : () -> ()
}) : () -> ()


