"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 5 : index} : () -> index
    %2 = "async.create_group"(%1) : (index) -> !async.group
    %3 = "async.execute"() ({
      "async.yield"() : () -> ()
    }) {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> !async.token
    %4 = "async.execute"() ({
      "async.yield"() : () -> ()
    }) {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> !async.token
    %5 = "async.execute"() ({
      "async.yield"() : () -> ()
    }) {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> !async.token
    %6 = "async.execute"() ({
      "async.yield"() : () -> ()
    }) {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> !async.token
    %7 = "async.execute"() ({
      "async.yield"() : () -> ()
    }) {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> !async.token
    %8 = "async.add_to_group"(%3, %2) : (!async.token, !async.group) -> index
    %9 = "async.add_to_group"(%4, %2) : (!async.token, !async.group) -> index
    %10 = "async.add_to_group"(%5, %2) : (!async.token, !async.group) -> index
    %11 = "async.add_to_group"(%6, %2) : (!async.token, !async.group) -> index
    %12 = "async.add_to_group"(%7, %2) : (!async.token, !async.group) -> index
    %13 = "async.execute"() ({
      "async.await_all"(%2) : (!async.group) -> ()
      "async.yield"() : () -> ()
    }) {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> !async.token
    %14 = "async.create_group"(%0) : (index) -> !async.group
    %15 = "async.add_to_group"(%13, %14) : (!async.token, !async.group) -> index
    "async.await_all"(%14) : (!async.group) -> ()
    "func.call"() {callee = @mlirAsyncRuntimePrintCurrentThreadId} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "mlirAsyncRuntimePrintCurrentThreadId", sym_visibility = "private"} : () -> ()
}) : () -> ()

// -----
