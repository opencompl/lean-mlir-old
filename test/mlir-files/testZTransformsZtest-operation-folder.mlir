"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 42 : i32} : () -> i32
    %1 = "test.op_in_place_fold_anchor"(%0) : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "foo"} : () -> ()
  "func.func"() ({
    %0 = "test.cast"() {test_fold_before_previously_folded_op} : () -> i32
    %1 = "test.cast"() {test_fold_before_previously_folded_op} : () -> i32
    "func.return"(%0, %1) : (i32, i32) -> ()
  }) {function_type = () -> (i32, i32), sym_name = "test_fold_before_previously_folded_op"} : () -> ()
}) : () -> ()

// -----
