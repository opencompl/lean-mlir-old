"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = true} : () -> i1
    %1 = "arith.constant"() {value = 1 : i32} : () -> i32
    %2 = "arith.select"(%0, %1, %arg0) : (i1, i32, i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "no_control_flow"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = true} : () -> i1
    %1 = "arith.constant"() {value = 1 : i32} : () -> i32
    "cf.cond_br"(%0, %arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3 × i32>} : (i1, i32) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%1)[^bb2] : (i32) -> ()
  ^bb2(%2: i32):  // 2 preds: ^bb0, ^bb1
    "func.return"(%2) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "simple_control_flow"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i1):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    "cf.cond_br"(%arg1, %arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3 × i32>} : (i1, i32) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%0)[^bb2] : (i32) -> ()
  ^bb2(%1: i32):  // 2 preds: ^bb0, ^bb1
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32, i1) -> i32, sym_name = "simple_control_flow_overdefined"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i1):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.constant"() {value = 2 : i32} : () -> i32
    "cf.cond_br"(%arg1, %arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3 × i32>} : (i1, i32) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%1)[^bb2] : (i32) -> ()
  ^bb2(%2: i32):  // 2 preds: ^bb0, ^bb1
    "func.return"(%2) : (i32) -> ()
  }) {function_type = (i32, i1) -> i32, sym_name = "simple_control_flow_constant_overdefined"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i1):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    "foo.cond_br"()[^bb1, ^bb2] : () -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%0)[^bb2] : (i32) -> ()
  ^bb2(%1: i32):  // 2 preds: ^bb0, ^bb1
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32, i1) -> i32, sym_name = "unknown_terminator"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> i1, sym_name = "ext_cond_fn", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i1):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    "cf.cond_br"(%arg1, %0, %0)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, i32, i32) -> ()
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb1
    %2 = "arith.constant"() {value = 0 : i32} : () -> i32
    %3 = "arith.addi"(%1, %2) : (i32, i32) -> i32
    %4 = "func.call"() {callee = @ext_cond_fn} : () -> i1
    "cf.cond_br"(%4, %3, %3)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, i32, i32) -> ()
  ^bb2(%5: i32):  // 2 preds: ^bb0, ^bb1
    "func.return"(%5) : (i32) -> ()
  }) {function_type = (i32, i1) -> i32, sym_name = "simple_loop"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    "cf.br"(%0)[^bb1] : (i32) -> ()
  ^bb1(%1: i32):  // 3 preds: ^bb0, ^bb3, ^bb4
    %2 = "func.call"() {callee = @ext_cond_fn} : () -> i1
    "cf.cond_br"(%2, %1)[^bb5, ^bb2] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3 × i32>} : (i1, i32) -> ()
  ^bb2:  // pred: ^bb1
    %3 = "arith.constant"() {value = 20 : i32} : () -> i32
    %4 = "arith.cmpi"(%1, %3) {predicate = 6 : i64} : (i32, i32) -> i1
    "cf.cond_br"(%4)[^bb3, ^bb4] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb3:  // pred: ^bb2
    %5 = "arith.constant"() {value = 1 : i32} : () -> i32
    "cf.br"(%5)[^bb1] : (i32) -> ()
  ^bb4:  // pred: ^bb2
    %6 = "arith.addi"(%1, %0) : (i32, i32) -> i32
    "cf.br"(%6)[^bb1] : (i32) -> ()
  ^bb5(%7: i32):  // pred: ^bb1
    "func.return"(%7) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "simple_loop_inner_control_flow"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (i1, i32), sym_name = "ext_cond_and_value_fn", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i1):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    "cf.cond_br"(%arg1, %0, %0)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, i32, i32) -> ()
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb1
    %2:2 = "func.call"() {callee = @ext_cond_and_value_fn} : () -> (i1, i32)
    "cf.cond_br"(%2#0, %2#1, %2#1)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, i32, i32) -> ()
  ^bb2(%3: i32):  // 2 preds: ^bb0, ^bb1
    "func.return"(%3) : (i32) -> ()
  }) {function_type = (i32, i1) -> i32, sym_name = "simple_loop_overdefined"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    %0 = "arith.constant"() {value = true} : () -> i1
    %1 = "arith.constant"() {value = false} : () -> i1
    "cf.cond_br"(%arg0, %1)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3 × i32>} : (i1, i1) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%0)[^bb2] : (i1) -> ()
  ^bb2(%2: i1):  // 2 preds: ^bb0, ^bb1
    "cf.br"(%2)[^bb3] : (i1) -> ()
  ^bb3(%3: i1):  // pred: ^bb2
    "func.return"(%2, %3) : (i1, i1) -> ()
  }) {function_type = (i1) -> (i1, i1), sym_name = "recheck_executable_edge"} : () -> ()
}) : () -> ()

// -----
