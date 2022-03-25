"builtin.module"() ({
  %0 = "foo.op"() : () -> i64
  %1 = "foo.op"() : () -> i32
  %2 = "foo.op"() : () -> memref<1xf64>
  "test.format_literal_op"() {foo.some_attr} : () -> ()
  "test.format_attr_op"() {attr = 10 : i64} : () -> ()
  "test.format_opt_attr_op_a"() {opt_attr = 10 : i64} : () -> ()
  "test.format_opt_attr_op_a"() : () -> ()
  "test.format_opt_attr_op_b"() {opt_attr = 10 : i64} : () -> ()
  "test.format_opt_attr_op_b"() : () -> ()
  "test.format_symbol_name_attr_op"() {attr = "name"} : () -> ()
  "test.format_symbol_name_attr_op"() {attr = "opt_name"} : () -> ()
  "test.format_opt_symbol_name_attr_op"() : () -> ()
  "test.format_attr_dict_w_keyword"() {attr = 10 : i64} : () -> ()
  "test.format_attr_dict_w_keyword"() {attr = 10 : i64, opt_attr = 10 : i64} : () -> ()
  %3 = "test.format_buildable_type_op"(%0) : (i64) -> i64
  "test.format_region_a_op"() ({
    "test.return"() : () -> ()
  }) : () -> ()
  "test.format_region_b_op"() ({
    "test.return"() : () -> ()
  }) : () -> ()
  "test.format_region_c_op"() ({
    "test.return"() : () -> ()
  }) : () -> ()
  "test.format_region_c_op"() ({
  }) : () -> ()
  "test.format_variadic_region_a_op"() ({
    "test.return"() : () -> ()
  }, {
    "test.return"() : () -> ()
  }) : () -> ()
  "test.format_variadic_region_b_op"() ({
    "test.return"() : () -> ()
  }, {
    "test.return"() : () -> ()
  }) : () -> ()
  "test.format_variadic_region_b_op"() : () -> ()
  "test.format_implicit_terminator_region_a_op"() ({
    "test.return"() : () -> ()
  }) : () -> ()
  "test.format_implicit_terminator_region_a_op"() ({
    "test.return"() {foo.attr} : () -> ()
  }) : () -> ()
  "test.format_implicit_terminator_region_a_op"() ({
    "test.return"(%0) : (i64) -> ()
  }) : () -> ()
  %4:2 = "test.format_result_a_op"() : () -> (i64, memref<1xf64>)
  %5:2 = "test.format_result_b_op"() : () -> (i64, memref<1xf64>)
  %6:2 = "test.format_result_c_op"() : () -> (i64, memref<1xf64>)
  %7:3 = "test.format_variadic_result"() : () -> (i64, i64, i64)
  %8:5 = "test.format_multiple_variadic_results"() {result_segment_sizes = dense<[3, 2]> : vector<2xi32>} : () -> (i64, i64, i64, i32, i32)
  "test.format_operand_a_op"(%0, %2) : (i64, memref<1xf64>) -> ()
  "test.format_operand_b_op"(%0, %2) : (i64, memref<1xf64>) -> ()
  "test.format_operand_c_op"(%0, %2) : (i64, memref<1xf64>) -> ()
  "test.format_operand_d_op"(%0, %2) : (i64, memref<1xf64>) -> ()
  "test.format_operand_e_op"(%0, %2) : (i64, memref<1xf64>) -> ()
  "test.format_variadic_operand"(%0, %0, %0) : (i64, i64, i64) -> ()
  "test.format_variadic_of_variadic_operand"(%0, %0, %0) {operand_segments = dense<[2, 0, 1]> : tensor<3xi32>} : (i64, i64, i64) -> ()
  "test.format_multiple_variadic_operands"(%0, %0, %0, %0, %1) {operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (i64, i64, i64, i64, i32) -> ()
  "foo.successor_test_region"() ({
    "test.format_successor_a_op"()[^bb1] {attr} : () -> ()
  ^bb1:  // 2 preds: ^bb0, ^bb1
    "test.format_successor_a_op"()[^bb1, ^bb2] {attr} : () -> ()
  ^bb2:  // pred: ^bb1
    "test.format_successor_a_op"() {attr} : () -> ()
  }) {arg_names = ["i", "j", "k"]} : () -> ()
  "test.format_optional_unit_attribute"() {is_optional} : () -> ()
  "test.format_optional_unit_attribute"() : () -> ()
  "test.format_optional_unit_attribute_no_elide"() {is_optional} : () -> ()
  "test.format_optional_enum_attr"() {attr = 5 : i64} : () -> ()
  "test.format_optional_enum_attr"() : () -> ()
  %9 = "test.format_optional_operand_result_a_op"(%0) {operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (i64) -> i64
  %10 = "test.format_optional_operand_result_a_op"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> i64
  "test.format_optional_operand_result_a_op"(%0) {operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (i64) -> ()
  "test.format_optional_operand_result_a_op"(%0, %0, %0) {operand_segment_sizes = dense<[1, 2]> : vector<2xi32>} : (i64, i64, i64) -> ()
  %11 = "test.format_optional_operand_result_b_op"(%0) {operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (i64) -> i64
  %12 = "test.format_optional_operand_result_b_op"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> i64
  %13 = "test.format_optional_operand_result_b_op"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> i64
  "test.format_optional_result_a_op"() {result_segment_sizes = dense<0> : vector<2xi32>} : () -> ()
  %14:3 = "test.format_optional_result_a_op"() {result_segment_sizes = dense<[1, 2]> : vector<2xi32>} : () -> (i64, i64, i64)
  "test.format_optional_result_b_op"() {result_segment_sizes = dense<0> : vector<2xi32>} : () -> ()
  %15:3 = "test.format_optional_result_b_op"() {result_segment_sizes = dense<[1, 2]> : vector<2xi32>} : () -> (i64, i64, i64)
  %16:3 = "test.format_optional_result_c_op"() {result_segment_sizes = dense<[1, 2]> : vector<2xi32>} : () -> (i64, i64, i64)
  "test.format_optional_else"() {isFirstBranchPresent} : () -> ()
  "test.format_optional_else"() : () -> ()
  "test.format_compound_attr"() {compound = #test.cmpnd_a<1, !test.smpla, [5, 6]>} : () -> ()
  "builtin.module"() ({
  ^bb0:
  }) {test.nested = #test.cmpnd_nested<nested = <1, !test.smpla, [5, 6]>>} : () -> ()
  "builtin.module"() ({
  ^bb0:
  }) {test.nested = #test.cmpnd_nested<nested = <1, !test.smpla, [5, 6]>>} : () -> ()
  "test.format_nested_attr"() {nested = #test.cmpnd_nested<nested = <1, !test.smpla, [5, 6]>>} : () -> ()
  "test.format_nested_attr"() {nested = #test.cmpnd_nested<nested = <1, !test.smpla, [5, 6]>>} : () -> ()
  "builtin.module"() ({
  ^bb0:
  }) {test.someAttr = #test.cmpnd_nested_inner<42 <1, !test.smpla, [5, 6]>>} : () -> ()
  "builtin.module"() ({
  ^bb0:
  }) {test.someAttr = #test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>} : () -> ()
  "test.format_cpmd_nested_attr"() {nested = #test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>} : () -> ()
  "test.format_qual_cpmd_nested_attr"() {nested = #test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>):
    "test.format_qual_cpmd_nested_type"(%arg0) : (!test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>) -> (), sym_name = "qualifiedCompoundNestedExplicit"} : () -> ()
  "test.format_custom_directive_operands"(%0, %0, %0) {operand_segment_sizes = dense<1> : vector<3xi32>} : (i64, i64, i64) -> ()
  "test.format_custom_directive_operands"(%0, %0) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : (i64, i64) -> ()
  "test.format_custom_directive_operands_and_types"(%0, %0, %0) {operand_segment_sizes = dense<1> : vector<3xi32>} : (i64, i64, i64) -> ()
  "test.format_custom_directive_operands_and_types"(%0, %0) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : (i64, i64) -> ()
  "test.format_custom_directive_attributes"() {attr = 54 : i64} : () -> ()
  "test.format_custom_directive_attributes"() {attr = 54 : i64, optAttr = 46 : i64} : () -> ()
  "test.format_custom_directive_regions"() ({
    "test.return"() : () -> ()
  }) : () -> ()
  "test.format_custom_directive_regions"() ({
    "test.return"() : () -> ()
  }, {
    "test.return"() : () -> ()
  }) : () -> ()
  %17:3 = "test.format_custom_directive_results"() {result_segment_sizes = dense<1> : vector<3xi32>} : () -> (i64, i64, i64)
  %18:2 = "test.format_custom_directive_results"() {result_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : () -> (i64, i64)
  %19:3 = "test.format_custom_directive_results_with_type_refs"() {result_segment_sizes = dense<1> : vector<3xi32>} : () -> (i64, i64, i64)
  %20:2 = "test.format_custom_directive_results_with_type_refs"() {result_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : () -> (i64, i64)
  "test.format_custom_directive_with_optional_operand_ref"(%0) : (i64) -> ()
  "test.format_custom_directive_with_optional_operand_ref"() : () -> ()
  "func.func"() ({
    "test.format_custom_directive_successors"()[^bb1, ^bb2, ^bb2] : () -> ()
  ^bb1:  // pred: ^bb0
    "test.format_custom_directive_successors"()[^bb2] : () -> ()
  ^bb2:  // 3 preds: ^bb0, ^bb0, ^bb1
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "foo"} : () -> ()
  %21 = "test.format_infer_variadic_type_from_non_variadic"(%0, %0) : (i64, i64) -> i64
  %22 = "test.format_all_types_match_var"(%0, %0) : (i64, i64) -> i64
  %23 = "test.format_all_types_match_attr"(%0) {value1 = 1 : i64} : (i64) -> i64
  %24 = "test.format_types_match_var"(%0) : (i64) -> i64
  %25:3 = "test.format_types_match_variadic"(%0, %0, %0) : (i64, i64, i64) -> (i64, i64, i64)
  %26 = "test.format_types_match_attr"() {value = 1 : i64} : () -> i64
  %27 = "test.format_types_match_context"(%0) : (i64) -> tuple<i64>
  %28 = "test.format_infer_type"() : () -> i16
  %29 = "test.format_infer_type2"() : () -> i16
  %30:2 = "test.format_infer_type_all_operands_and_types"(%0, %1) : (i64, i32) -> (i64, i32)
  %31:2 = "test.format_infer_type_all_types_one_operand"(%0, %1) : (i64, i32) -> (i64, i32)
  %32:4 = "test.format_infer_type_all_types_two_operands"(%0, %1, %0, %1) : (i64, i32, i64, i32) -> (i64, i32, i64, i32)
  %33:2 = "test.format_infer_type_all_types"(%0, %1) : (i64, i32) -> (i64, i32)
  %34:2 = "test.format_infer_type_regions"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.terminator"() : () -> ()
  }) : () -> (i32, f32)
  %35:4 = "test.format_infer_type_variadic_operands"(%1, %1, %0, %0) {operand_segment_sizes = dense<2> : vector<2xi32>} : (i32, i32, i64, i64) -> (i32, i32, i64, i64)
  "test.has_str_value"() : () -> ()
}) : () -> ()

// -----
