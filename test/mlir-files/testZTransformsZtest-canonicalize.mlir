"builtin.module"() ({
  "func.func"() ({
    "test.op_with_region_pattern"() ({
      "test.op_with_region_terminator"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "remove_op_with_inner_ops_pattern"} : () -> ()
  "func.func"() ({
    "test.op_with_region_fold_no_side_effect"() ({
      "test.op_with_region_terminator"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "remove_op_with_inner_ops_fold_no_side_effect"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_with_region_fold"(%arg0) ({
      "test.op_with_region_terminator"() : () -> ()
    }) : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "remove_op_with_inner_ops_fold"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0:2 = "test.op_with_variadic_results_and_folder"(%arg0, %arg1) : (i32, i32) -> (i32, i32)
    "func.return"(%0#0, %0#1) : (i32, i32) -> ()
  }) {function_type = (i32, i32) -> (i32, i32), sym_name = "remove_op_with_variadic_results_and_folder"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "arith.constant"() {value = 42 : i32} : () -> i32
    %1 = "arith.constant"() {value = 43 : i32} : () -> i32
    %2 = "test.op_commutative"(%0, %arg0, %arg1, %1) : (i32, i32, i32, i32) -> i32
    %3 = "test.op_commutative"(%arg0, %0, %1, %arg1) : (i32, i32, i32, i32) -> i32
    "func.return"(%2, %3) : (i32, i32) -> ()
  }) {function_type = (i32, i32) -> (i32, i32), sym_name = "test_commutative_multi"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "arith.constant"() {value = 42 : i32} : () -> i32
    %1 = "arith.constant"() {value = 42 : i32} : () -> i32
    %2 = "test.op_commutative"(%0, %arg0, %arg1, %1) : (i32, i32, i32, i32) -> i32
    %3 = "arith.constant"() {value = 42 : i32} : () -> i32
    %4 = "test.op_commutative"(%arg0, %3, %1, %arg1) : (i32, i32, i32, i32) -> i32
    "func.return"(%2, %4) : (i32, i32) -> ()
  }) {function_type = (i32, i32) -> (i32, i32), sym_name = "test_commutative_multi_cst"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 4.200000e+01 : f32} : () -> f32
    %1 = "test.passthrough_fold"(%0) : (f32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "typemismatch"} : () -> ()
  "func.func"() ({
    %0 = "test.dialect_canonicalizable"() : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "test_dialect_canonicalizer"} : () -> ()
}) : () -> ()

// -----
