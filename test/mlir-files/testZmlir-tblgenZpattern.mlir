"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_a"(%arg0) {attr = 10 : i32} : (i32) -> i32
    %1 = "test.op_a"(%0) {attr = 20 : i32} : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "verifyFusedLocs"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.loc_src"(%arg0) : (i32) -> i32
    %1 = "test.loc_src"(%0) : (i32) -> i32
    %2 = "test.loc_src"(%1) : (i32) -> i32
    "test.loc_src_no_res"(%2) : (i32) -> ()
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "verifyDesignatedLoc"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    "test.op_h"(%arg0) : (i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32) -> (), sym_name = "verifyZeroResult"} : () -> ()
  "func.func"() ({
    %0 = "test.op_j"() : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "verifyZeroArg"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f32):
    "test.ignore_arg_match_src"(%arg0, %arg1, %arg2) {d = 42 : i64, e = 24 : i64, f = 15 : i64} : (i32, i32, i32) -> ()
    "test.ignore_arg_match_src"(%arg0, %arg1, %arg3) {d = 42 : i64, e = 24 : i64, f = 15 : i64} : (i32, i32, f32) -> ()
    "test.ignore_arg_match_src"(%arg0, %arg1, %arg2) {d = 42 : i32, e = 24 : i64, f = 15 : i64} : (i32, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32, i32, i32, f32) -> (), sym_name = "testIgnoreArgMatch"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    "test.interleaved_operand_attr1"(%arg0, %arg1) {attr1 = 15 : i64, attr2 = 42 : i64} : (i32, i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32, i32) -> (), sym_name = "verifyInterleavedOperandAttribute"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_d"(%arg0) : (i32) -> i32
    %1 = "test.op_g"(%arg0) : (i32) -> i32
    %2 = "test.op_g"(%1) : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "verifyBenefit"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "test.native_code_call1"(%arg0, %arg1) {attr1 = 42 : i64, attr2 = 24 : i64, choice = true} : (i32, i32) -> i32
    %1 = "test.native_code_call1"(%arg0, %arg1) {attr1 = 42 : i64, attr2 = 24 : i64, choice = false} : (i32, i32) -> i32
    "func.return"(%0, %1) : (i32, i32) -> ()
  }) {function_type = (i32, i32) -> (i32, i32), sym_name = "verifyNativeCodeCall"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.native_code_call3"(%arg0) : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "verifyAuxiliaryNativeCodeCall"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_k"() : () -> i32
    %1:2 = "test.native_code_call4"(%0) : (i32) -> (i32, i32)
    %2 = "test.constant"() {value = 1 : i8} : () -> i8
    %3:2 = "test.native_code_call4"(%2) : (i8) -> (i32, i32)
    "func.return"(%1#0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "verifyNativeCodeCallBinding"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_k"() : () -> i32
    %1 = "test.op_k"() : () -> i32
    %2:2 = "test.native_code_call6"(%0, %1) : (i32, i32) -> (i32, i32)
    "func.return"(%2#0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "verifyMultipleNativeCodeCallBinding"} : () -> ()
  "func.func"() ({
    %0 = "test.all_attr_constraint_of1"() {attr = [0, 1]} : () -> i32
    %1 = "test.all_attr_constraint_of1"() {attr = [0, 2]} : () -> i32
    %2 = "test.all_attr_constraint_of1"() {attr = [-1, 1]} : () -> i32
    "func.return"(%0, %1, %2) : (i32, i32, i32) -> ()
  }) {function_type = () -> (i32, i32, i32), sym_name = "verifyAllAttrConstraintOf"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    "test.many_arguments"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0) {attr1 = 42 : i64, attr2 = 42 : i64, attr3 = 42 : i64, attr4 = 42 : i64, attr5 = 42 : i64, attr6 = 42 : i64, attr7 = 42 : i64, attr8 = 42 : i64, attr9 = 42 : i64} : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32) -> (), sym_name = "verifyManyArgs"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "test.op_n"(%arg0, %arg0) : (i32, i32) -> i32
    %1 = "test.op_n"(%arg0, %arg1) : (i32, i32) -> i32
    "func.return"() : () -> ()
  }) {function_type = (i32, i32) -> (), sym_name = "verifyEqualArgs"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32):
    %0 = "test.op_p"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (i32, i32, i32, i32, i32, i32) -> i32
    %1 = "test.op_n"(%arg1, %0) : (i32, i32) -> i32
    %2 = "test.op_p"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (i32, i32, i32, i32, i32, i32) -> i32
    %3 = "test.op_n"(%arg0, %2) : (i32, i32) -> i32
    "func.return"() : () -> ()
  }) {function_type = (i32, i32, i32, i32, i32, i32) -> (), sym_name = "verifyNestedOpEqualArgs"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "test.op_n"(%arg1, %arg0) : (i32, i32) -> i32
    %1 = "test.op_n"(%0, %arg0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32, i32) -> i32, sym_name = "verifyNestedSameOpAndSameArgEquality"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32):
    %0 = "test.op_p"(%arg0, %arg1, %arg0, %arg0, %arg1, %arg2) : (i32, i32, i32, i32, i32, i32) -> i32
    %1 = "test.op_p"(%arg0, %arg1, %arg0, %arg0, %arg0, %arg2) : (i32, i32, i32, i32, i32, i32) -> i32
    %2 = "test.op_p"(%arg0, %arg1, %arg1, %arg0, %arg1, %arg2) : (i32, i32, i32, i32, i32, i32) -> i32
    %3 = "test.op_p"(%arg0, %arg1, %arg2, %arg2, %arg3, %arg4) : (i32, i32, i32, i32, i32, i32) -> i32
    "func.return"() : () -> ()
  }) {function_type = (i32, i32, i32, i32, i32) -> (), sym_name = "verifyMultipleEqualArgs"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.symbol_binding_a"(%arg0) {attr = 42 : i64} : (i32) -> i32
    %1 = "test.symbol_binding_a"(%arg0) {attr = 42 : i64} : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "symbolBinding"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    "test.symbol_binding_no_result"(%arg0) : (i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32) -> (), sym_name = "symbolBindingNoResult"} : () -> ()
  "func.func"() ({
    %0 = "test.match_op_attribute1"() {default_valued_attr = 3 : i32, more_attr = 4 : i32, optional_attr = 2 : i32, required_attr = 1 : i32} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "succeedMatchOpAttr"} : () -> ()
  "func.func"() ({
    %0 = "test.match_op_attribute1"() {default_valued_attr = 3 : i32, more_attr = 4 : i32, required_attr = 1 : i32} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "succeedMatchMissingOptionalAttr"} : () -> ()
  "func.func"() ({
    %0 = "test.match_op_attribute1"() {more_attr = 4 : i32, optional_attr = 2 : i32, required_attr = 1 : i32} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "succeedMatchMissingDefaultValuedAttr"} : () -> ()
  "func.func"() ({
    %0 = "test.match_op_attribute1"() {more_attr = 5 : i32, optional_attr = 2 : i32, required_attr = 1 : i32} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "failedMatchAdditionalConstraintNotSatisfied"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_c"(%arg0) : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "verifyConstantAttr"} : () -> ()
  "func.func"() ({
    %0 = "test.match_op_attribute3"() {attr} : () -> i32
    %1 = "test.match_op_attribute3"() : () -> i32
    "func.return"(%0, %1) : (i32, i32) -> ()
  }) {function_type = () -> (i32, i32), sym_name = "verifyUnitAttr"} : () -> ()
  "func.func"() ({
    %0 = "test.constant"() {value = 1 : i32} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "testConstOp"} : () -> ()
  "func.func"() ({
    %0 = "test.constant"() {value = 1 : i32} : () -> i32
    %1 = "test.op_s"(%0) {value = 1 : i32} : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "testConstOpUsed"} : () -> ()
  "func.func"() ({
    %0 = "test.constant"() {value = 1 : i32} : () -> i32
    %1 = "test.constant"() {value = 2 : i32} : () -> i32
    %2 = "test.op_r"(%0, %1) : (i32, i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "testConstOpReplaced"} : () -> ()
  "func.func"() ({
    %0 = "test.constant"() {value = 1 : i64} : () -> i64
    %1 = "test.constant"() {value = 2 : i64} : () -> i64
    %2 = "test.op_r"(%0, %1) : (i64, i64) -> i64
    "func.return"(%2) : (i64) -> ()
  }) {function_type = () -> i64, sym_name = "testConstOpMatchFailure"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.constant"() {value = 1 : i32} : () -> i32
    %1 = "test.op_r"(%0, %arg0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "testConstOpMatchNonConst"} : () -> ()
  "func.func"() ({
    %0 = "test.str_enum_attr"() {attr = "A"} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "verifyStrEnumAttr"} : () -> ()
  "func.func"() ({
    %0 = "test.i32_enum_attr"() {attr = 5 : i32} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "verifyI32EnumAttr"} : () -> ()
  "func.func"() ({
    %0 = "test.i64_enum_attr"() {attr = 5 : i64} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "verifyI64EnumAttr"} : () -> ()
  "func.func"() ({
    "test.i32ElementsAttr"() {attr = dense<[3, 5]> : tensor<2 × i32>} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "rewrite_i32elementsattr"} : () -> ()
  "func.func"() ({
    "test.float_elements_attr"() {scalar_f32_attr = dense<[3.000000e+00, 4.000000e+00]> : tensor<2 × f32>, tensor_f64_attr = dense<6.000000e+00> : tensor<4x8xf64>} : () -> ()
    "test.float_elements_attr"() {scalar_f32_attr = dense<7.000000e+00> : tensor<2 × f32>, tensor_f64_attr = dense<3.000000e+00> : tensor<4x8xf64>} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "rewrite_f64elementsattr"} : () -> ()
  "func.func"() ({
    %0:3 = "test.three_result"() {kind = 1 : i64} : () -> (i32, f32, f32)
    "func.return"(%0#0, %0#1, %0#2) : (i32, f32, f32) -> ()
  }) {function_type = () -> (i32, f32, f32), sym_name = "useMultiResultOpToReplaceWhole"} : () -> ()
  "func.func"() ({
    %0:3 = "test.three_result"() {kind = 2 : i64} : () -> (i32, f32, f32)
    "func.return"(%0#0, %0#1, %0#2) : (i32, f32, f32) -> ()
  }) {function_type = () -> (i32, f32, f32), sym_name = "useMultiResultOpToReplacePartial1"} : () -> ()
  "func.func"() ({
    %0:3 = "test.three_result"() {kind = 3 : i64} : () -> (i32, f32, f32)
    "func.return"(%0#0, %0#1, %0#2) : (i32, f32, f32) -> ()
  }) {function_type = () -> (i32, f32, f32), sym_name = "useMultiResultOpToReplacePartial2"} : () -> ()
  "func.func"() ({
    %0:3 = "test.three_result"() {kind = 4 : i64} : () -> (i32, f32, f32)
    "func.return"(%0#0, %0#1, %0#2) : (i32, f32, f32) -> ()
  }) {function_type = () -> (i32, f32, f32), sym_name = "useMultiResultOpResultsSeparately"} : () -> ()
  "func.func"() ({
    %0:2 = "test.two_result"() {kind = 5 : i64} : () -> (i32, f32)
    %1:2 = "test.two_result"() {kind = 5 : i64} : () -> (i32, f32)
    "func.return"(%0#0, %0#1, %1#0) : (i32, f32, i32) -> ()
  }) {function_type = () -> (i32, f32, i32), sym_name = "constraintOnSourceOpResult"} : () -> ()
  "func.func"() ({
    %0:3 = "test.three_result"() {kind = 6 : i64} : () -> (i32, f32, f32)
    "func.return"(%0#0, %0#1, %0#2) : (i32, f32, f32) -> ()
  }) {function_type = () -> (i32, f32, f32), sym_name = "useAuxiliaryOpToReplaceMultiResultOp"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    %0 = "test.one_variadic_out_one_variadic_in1"(%arg0) : (i32) -> i32
    %1:2 = "test.one_variadic_out_one_variadic_in1"(%arg0, %arg1) : (i32, i32) -> (i32, i32)
    %2:3 = "test.one_variadic_out_one_variadic_in1"(%arg0, %arg1, %arg2) : (i32, i32, i32) -> (i32, i32, i32)
    "func.return"(%0, %1#0, %1#1, %2#0, %2#1, %2#2) : (i32, i32, i32, i32, i32, i32) -> ()
  }) {function_type = (i32, i32, i32) -> (i32, i32, i32, i32, i32, i32), sym_name = "replaceOneVariadicOutOneVariadicInOp"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: f32, %arg2: i32):
    "test.mixed_variadic_in1"(%arg1) : (f32) -> ()
    "test.mixed_variadic_in1"(%arg0, %arg1, %arg2) : (i32, f32, i32) -> ()
    "test.mixed_variadic_in1"(%arg0, %arg0, %arg1, %arg2, %arg2) : (i32, i32, f32, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i32, f32, i32) -> (), sym_name = "replaceMixedVariadicInputOp"} : () -> ()
  "func.func"() ({
    %0 = "test.mixed_variadic_out1"() : () -> f32
    %1:3 = "test.mixed_variadic_out1"() : () -> (i32, f32, i32)
    %2:5 = "test.mixed_variadic_out1"() : () -> (i32, i32, f32, i32, i32)
    "func.return"(%0, %1#0, %1#1, %1#2, %2#0, %2#1, %2#2, %2#3, %2#4) : (f32, i32, f32, i32, i32, i32, f32, i32, i32) -> ()
  }) {function_type = () -> (f32, i32, f32, i32, i32, i32, f32, i32, i32), sym_name = "replaceMixedVariadicOutputOp"} : () -> ()
  "func.func"() ({
    %0 = "test.one_i32_out"() : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "generateVariadicOutputOpInNestedPattern"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_m"(%arg0) : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "redundantTest"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i16, %arg2: i8):
    %0 = "test.either_op_a"(%arg0, %arg1, %arg2) : (i32, i16, i8) -> i32
    %1 = "test.either_op_a"(%arg1, %arg0, %arg2) : (i16, i32, i8) -> i32
    "func.return"() : () -> ()
  }) {function_type = (i32, i16, i8) -> (), sym_name = "either_dag_leaf_only_1"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i16, %arg2: i8):
    %0 = "test.either_op_b"(%arg0) : (i32) -> i32
    %1 = "test.either_op_a"(%0, %arg1, %arg2) : (i32, i16, i8) -> i32
    %2 = "test.either_op_a"(%arg1, %0, %arg2) : (i16, i32, i8) -> i32
    "func.return"() : () -> ()
  }) {function_type = (i32, i16, i8) -> (), sym_name = "either_dag_leaf_dag_node"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i16, %arg2: i8):
    %0 = "test.either_op_b"(%arg0) : (i32) -> i32
    %1 = "test.either_op_b"(%arg1) : (i16) -> i32
    %2 = "test.either_op_a"(%0, %1, %arg2) : (i32, i32, i8) -> i32
    %3 = "test.either_op_a"(%1, %0, %arg2) : (i32, i32, i8) -> i32
    "func.return"() : () -> ()
  }) {function_type = (i32, i16, i8) -> (), sym_name = "either_dag_node_dag_node"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i64):
    %0 = "test.source_op"(%arg0) {tag = 11 : i32} : (i64) -> i8
    "func.return"(%0) : (i8) -> ()
  }) {function_type = (i64) -> i8, sym_name = "explicitReturnTypeTest"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    %0 = "test.source_op"(%arg0) {tag = 22 : i32} : (i1) -> i8
    "func.return"(%0) : (i8) -> ()
  }) {function_type = (i1) -> i8, sym_name = "returnTypeBuilderTest"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    %0 = "test.source_op"(%arg0) {tag = 33 : i32} : (i1) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (i1) -> i1, sym_name = "multipleReturnTypeBuildTest"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i8):
    %0 = "test.source_op"(%arg0) {tag = 44 : i32} : (i8) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i8) -> i32, sym_name = "copyValueType"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    %0 = "test.source_op"(%arg0) {tag = 55 : i32} : (i1) -> i64
    "func.return"(%0) : (i64) -> ()
  }) {function_type = (i1) -> i64, sym_name = "multipleReturnTypeDifferent"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.source_op"(%arg0) {tag = 66 : i32} : (i32) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (i32) -> i1, sym_name = "returnTypeAndLocation"} : () -> ()
  "func.func"() ({
    "test.no_str_value"() {value = "bar"} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "testConstantStrAttr"} : () -> ()
}) : () -> ()

// -----
