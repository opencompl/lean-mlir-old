"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_involution_trait_no_operation_fold"(%arg0) : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "testSingleInvolution"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_involution_trait_no_operation_fold"(%arg0) : (i32) -> i32
    %1 = "test.op_involution_trait_no_operation_fold"(%0) : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "testDoubleInvolution"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_involution_trait_no_operation_fold"(%arg0) : (i32) -> i32
    %1 = "test.op_involution_trait_no_operation_fold"(%0) : (i32) -> i32
    %2 = "test.op_involution_trait_no_operation_fold"(%1) : (i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "testTripleInvolution"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_involution_trait_failing_operation_fold"(%arg0) : (i32) -> i32
    %1 = "test.op_involution_trait_failing_operation_fold"(%0) : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "testFailingOperationFolder"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_involution_trait_succesful_operation_fold"(%arg0) : (i32) -> i32
    %1 = "test.op_involution_trait_succesful_operation_fold"(%0) : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "testInhibitInvolution"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_idempotent_trait"(%arg0) : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "testSingleIdempotent"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_idempotent_trait"(%arg0) : (i32) -> i32
    %1 = "test.op_idempotent_trait"(%0) : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "testDoubleIdempotent"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_idempotent_trait"(%arg0) : (i32) -> i32
    %1 = "test.op_idempotent_trait"(%0) : (i32) -> i32
    %2 = "test.op_idempotent_trait"(%1) : (i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "testTripleIdempotent"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_idempotent_trait_binary"(%arg0, %arg0) : (i32, i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "testBinaryIdempotent"} : () -> ()
}) : () -> ()


