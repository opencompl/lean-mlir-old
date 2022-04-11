"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 43 : i32} : () -> i32
    %1 = "test.op_commutative2"(%0, %arg0) : (i32, i32) -> i32
    %2 = "test.op_commutative2"(%1, %arg0) : (i32, i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "test_reorder_constants_and_match"} : () -> ()
}) : () -> ()


