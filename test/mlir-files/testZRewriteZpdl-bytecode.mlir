"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.apply_constraint"(%arg0, %arg0)[^bb1, ^bb3] {name = "multi_entity_constraint"} : (!pdl.operation, !pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.apply_constraint"(%arg0)[^bb2, ^bb3] {name = "single_entity_constraint"} : (!pdl.operation) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.replaced_by_pattern", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() {test_attr} : () -> ()
  }) {sym_name = "ir", test.apply_constraint_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_results"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      %1 = "pdl_interp.get_value_type"(%0) : (!pdl.range<value>) -> !pdl.range<type>
      "pdl_interp.apply_constraint"(%0, %1)[^bb1, ^bb2] {name = "multi_entity_var_constraint"} : (!pdl.range<value>, !pdl.range<type>) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.replaced_by_pattern", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.failure_op"() {test_attr} : () -> ()
    %0:2 = "test.success_op"() : () -> (i32, i64)
  }) {sym_name = "ir", test.apply_constraint_2} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb2] {name = "test.op"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.get_operand"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
        "pdl_interp.apply_rewrite"(%arg0, %0) {name = "rewriter"} : (!pdl.operation, !pdl.value) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op_input"() : () -> i32
    "test.op"(%0) : (i32) -> ()
  }) {sym_name = "ir", test.apply_rewrite_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb2] {name = "test.op"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.apply_rewrite"(%arg0) {name = "creator"} : (!pdl.operation) -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() : () -> ()
  }) {sym_name = "ir", test.apply_rewrite_2} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb2] {name = "test.op"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0:2 = "pdl_interp.apply_rewrite"(%arg0) {name = "var_creator"} : (!pdl.operation) -> (!pdl.range<value>, !pdl.range<type>)
        %1 = "pdl_interp.create_operation"(%0#0, %0#1) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[1, 0, 1]> : vector<3 × i32>} : (!pdl.range<value>, !pdl.range<type>) -> !pdl.operation
        "pdl_interp.replace"(%arg0, %0#0) : (!pdl.operation, !pdl.range<value>) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.producer"() : () -> i32
    %1 = "test.op"(%0) : (i32) -> i32
    "test.consumer"(%1) : (i32) -> ()
  }) {sym_name = "ir", test.apply_rewrite_3} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb2] {name = "test.op"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.apply_rewrite"() {name = "type_creator"} : () -> !pdl.type
        %1 = "pdl_interp.create_operation"(%0) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[0, 0, 1]> : vector<3 × i32>} : (!pdl.type) -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() : () -> ()
  }) {sym_name = "ir", test.apply_rewrite_4} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.create_attribute"() {value} : () -> !pdl.attribute
      %1 = "pdl_interp.get_attribute"(%arg0) {name = "test_attr"} : (!pdl.operation) -> !pdl.attribute
      "pdl_interp.are_equal"(%0, %1)[^bb1, ^bb2] : (!pdl.attribute, !pdl.attribute) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() {test_attr} : () -> ()
  }) {sym_name = "ir", test.are_equal_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.create_types"() {value = [i32, i64]} : () -> !pdl.range<type>
      %1 = "pdl_interp.get_results"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      %2 = "pdl_interp.get_value_type"(%1) : (!pdl.range<value>) -> !pdl.range<type>
      "pdl_interp.are_equal"(%2, %0)[^bb1, ^bb2] : (!pdl.range<type>, !pdl.range<type>) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.not_equal"() : () -> i32
    %1:2 = "test.op"() : () -> (i32, i64)
  }) {sym_name = "ir", test.are_equal_2} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb3] {name = "test.op"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.branch"()[^bb2] : () -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 2 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 2 preds: ^bb0, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() : () -> ()
  }) {sym_name = "ir", test.branch_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_attribute"(%arg0) {name = "test_attr"} : (!pdl.operation) -> !pdl.attribute
      "pdl_interp.check_attribute"(%0)[^bb1, ^bb2] {constantValue} : (!pdl.attribute) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() {test_attr} : () -> ()
  }) {sym_name = "ir", test.check_attribute_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operand_count"(%arg0)[^bb1, ^bb3] {compareAtLeast, count = 1 : i32} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.check_operand_count"(%arg0)[^bb2, ^bb3] {count = 2 : i32} : (!pdl.operation) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op"() : () -> i32
    "test.op"(%0, %0) : (i32, i32) -> ()
  }) {sym_name = "ir", test.check_operand_count_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb2] {name = "test.op"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() : () -> ()
  }) {sym_name = "ir", test.check_operation_name_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_result_count"(%arg0)[^bb1, ^bb3] {compareAtLeast, count = 1 : i32} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.check_result_count"(%arg0)[^bb2, ^bb3] {count = 2 : i32} : (!pdl.operation) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op"() : () -> i32
    %1:2 = "test.op"() : () -> (i32, i32)
  }) {sym_name = "ir", test.check_result_count_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_attribute"(%arg0) {name = "test_attr"} : (!pdl.operation) -> !pdl.attribute
      "pdl_interp.is_not_null"(%0)[^bb1, ^bb3] : (!pdl.attribute) -> ()
    ^bb1:  // pred: ^bb0
      %1 = "pdl_interp.get_attribute_type"(%0) : (!pdl.attribute) -> !pdl.type
      "pdl_interp.check_type"(%1)[^bb2, ^bb3] {type = i32} : (!pdl.type) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() {test_attr = 10 : i32} : () -> ()
  }) {sym_name = "ir", test.check_type_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_results"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      %1 = "pdl_interp.get_value_type"(%0) : (!pdl.range<value>) -> !pdl.range<type>
      "pdl_interp.check_types"(%1)[^bb1, ^bb2] {types = [i32]} : (!pdl.range<type>) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0:2 = "test.op"() : () -> (i32, i64)
    %1 = "test.op"() : () -> i32
  }) {sym_name = "ir", test.check_types_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
^bb0:
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_attribute"(%arg0) {name = "test_attr"} : (!pdl.operation) -> !pdl.attribute
      "pdl_interp.is_not_null"(%0)[^bb1, ^bb3] : (!pdl.attribute) -> ()
    ^bb1:  // pred: ^bb0
      %1 = "pdl_interp.create_type"() {value = i32} : () -> !pdl.type
      %2 = "pdl_interp.get_attribute_type"(%0) : (!pdl.attribute) -> !pdl.type
      "pdl_interp.are_equal"(%2, %1)[^bb2, ^bb3] : (!pdl.type, !pdl.type) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() {test_attr = 0 : i32} : () -> ()
  }) {sym_name = "ir", test.create_type_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_result"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
      %1 = "pdl_interp.get_users"(%0) : (!pdl.value) -> !pdl.range<operation>
      %2 = "pdl_interp.extract"(%1) {index = 1 : i32} : (!pdl.range<operation>) -> !pdl.operation
      "pdl_interp.is_not_null"(%2)[^bb1, ^bb2] : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%2, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op"() : () -> i32
    %1 = "test.op"(%0) : (i32) -> i32
    %2 = "test.op"(%0, %0) : (i32, i32) -> i32
  }) {sym_name = "ir", test.extract_op} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_results"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      %1 = "pdl_interp.get_value_type"(%0) : (!pdl.range<value>) -> !pdl.range<type>
      %2 = "pdl_interp.extract"(%1) {index = 1 : i32} : (!pdl.range<type>) -> !pdl.type
      "pdl_interp.is_not_null"(%2)[^bb1, ^bb2] : (!pdl.type) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op"() : () -> i32
    %1:2 = "test.op"(%0) : (i32) -> (i32, i32)
    %2 = "test.op"(%0) : (i32) -> i32
  }) {sym_name = "ir", test.extract_type} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_results"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      %1 = "pdl_interp.extract"(%0) {index = 1 : i32} : (!pdl.range<value>) -> !pdl.value
      "pdl_interp.is_not_null"(%1)[^bb1, ^bb2] : (!pdl.value) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op"() : () -> i32
    %1:2 = "test.op"(%0) : (i32) -> (i32, i32)
    %2 = "test.op"(%0) : (i32) -> i32
  }) {sym_name = "ir", test.extract_value} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_result"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
      %1 = "pdl_interp.get_users"(%0) : (!pdl.value) -> !pdl.range<operation>
      "pdl_interp.foreach"(%1)[^bb1] ({
      ^bb0(%arg1: !pdl.operation):
        %2 = "pdl_interp.get_result"(%arg1) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
        %3 = "pdl_interp.get_users"(%2) : (!pdl.value) -> !pdl.range<operation>
        "pdl_interp.foreach"(%3)[^bb1] ({
        ^bb0(%arg2: !pdl.operation):
          "pdl_interp.record_match"(%arg2, %arg0)[^bb1] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
        ^bb1:  // pred: ^bb0
          "pdl_interp.continue"() : () -> ()
        }) : (!pdl.range<operation>) -> ()
      ^bb1:  // pred: ^bb0
        "pdl_interp.continue"() : () -> ()
      }) : (!pdl.range<operation>) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op"() : () -> i32
    %1 = "test.op"(%0) : (i32) -> i32
    %2 = "test.op"(%1) : (i32) -> i32
    %3 = "test.op"(%1) : (i32) -> i32
    %4 = "test.op"(%0) : (i32) -> i32
    %5 = "test.op"(%4) : (i32) -> i32
    %6 = "test.op"(%4) : (i32) -> i32
  }) {sym_name = "ir", test.foreach} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_result"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
      %1 = "pdl_interp.get_users"(%0) : (!pdl.value) -> !pdl.range<operation>
      "pdl_interp.foreach"(%1)[^bb1] ({
      ^bb0(%arg1: !pdl.operation):
        "pdl_interp.record_match"(%arg1, %arg0)[^bb1] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
      ^bb1:  // pred: ^bb0
        "pdl_interp.continue"() : () -> ()
      }) : (!pdl.range<operation>) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op"() : () -> i32
    %1 = "test.op"(%0) : (i32) -> i32
    %2 = "test.op"(%0, %0) : (i32, i32) -> i32
  }) {sym_name = "ir", test.get_users_of_value} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_result_count"(%arg0)[^bb1, ^bb2] {compareAtLeast, count = 2 : i32} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      %0 = "pdl_interp.get_results"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      %1 = "pdl_interp.get_users"(%0) : (!pdl.range<value>) -> !pdl.range<operation>
      "pdl_interp.foreach"(%1)[^bb2] ({
      ^bb0(%arg1: !pdl.operation):
        "pdl_interp.record_match"(%arg1, %arg0)[^bb1] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
      ^bb1:  // pred: ^bb0
        "pdl_interp.continue"() : () -> ()
      }) : (!pdl.range<operation>) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0:2 = "test.op"() : () -> (i32, i32)
    %1 = "test.op"(%0#0) : (i32) -> i32
    %2 = "test.op"(%0#1) : (i32) -> i32
  }) {sym_name = "ir", test.get_all_users_of_range} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_result_count"(%arg0)[^bb1, ^bb2] {compareAtLeast, count = 2 : i32} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      %0 = "pdl_interp.get_results"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      %1 = "pdl_interp.extract"(%0) {index = 0 : i32} : (!pdl.range<value>) -> !pdl.value
      %2 = "pdl_interp.get_users"(%1) : (!pdl.value) -> !pdl.range<operation>
      "pdl_interp.foreach"(%2)[^bb2] ({
      ^bb0(%arg1: !pdl.operation):
        "pdl_interp.record_match"(%arg1, %arg0)[^bb1] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
      ^bb1:  // pred: ^bb0
        "pdl_interp.continue"() : () -> ()
      }) : (!pdl.range<operation>) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0:2 = "test.op"() : () -> (i32, i32)
    %1 = "test.op"(%0#0) : (i32) -> i32
    %2 = "test.op"(%0#1) : (i32) -> i32
  }) {sym_name = "ir", test.get_first_users_of_range} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operand_count"(%arg0)[^bb1, ^bb3] {count = 5 : i32} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      %0 = "pdl_interp.get_operand"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
      %1 = "pdl_interp.get_operand"(%arg0) {index = 4 : i32} : (!pdl.operation) -> !pdl.value
      %2 = "pdl_interp.get_defining_op"(%0) : (!pdl.value) -> !pdl.operation
      %3 = "pdl_interp.get_defining_op"(%1) : (!pdl.value) -> !pdl.operation
      "pdl_interp.are_equal"(%2, %3)[^bb2, ^bb3] : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op"() : () -> i32
    %1 = "test.op"() : () -> i32
    "test.op"(%0, %0, %0, %0, %0) : (i32, i32, i32, i32, i32) -> ()
    "test.op"(%0, %0, %0, %0, %1) : (i32, i32, i32, i32, i32) -> ()
  }) {sym_name = "ir", test.get_defining_op_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operand_count"(%arg0)[^bb1, ^bb3] {count = 2 : i32} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      %0 = "pdl_interp.get_operands"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.range<value>
      %1 = "pdl_interp.get_operands"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      "pdl_interp.are_equal"(%0, %1)[^bb2, ^bb3] : (!pdl.range<value>, !pdl.range<value>) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.op"(%0#0, %0#1) : (i32, i32) -> ()
  }) {sym_name = "ir", test.get_operands_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb9] {name = "test.attr_sized_operands"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      %0 = "pdl_interp.get_operands"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.range<value>
      "pdl_interp.is_not_null"(%0)[^bb2, ^bb9] : (!pdl.range<value>) -> ()
    ^bb2:  // pred: ^bb1
      %1 = "pdl_interp.get_operands"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
      "pdl_interp.is_not_null"(%1)[^bb9, ^bb3] : (!pdl.value) -> ()
    ^bb3:  // pred: ^bb2
      %2 = "pdl_interp.get_operands"(%arg0) {index = 1 : i32} : (!pdl.operation) -> !pdl.range<value>
      "pdl_interp.is_not_null"(%2)[^bb4, ^bb9] : (!pdl.range<value>) -> ()
    ^bb4:  // pred: ^bb3
      %3 = "pdl_interp.get_operands"(%arg0) {index = 1 : i32} : (!pdl.operation) -> !pdl.value
      "pdl_interp.is_not_null"(%3)[^bb9, ^bb5] : (!pdl.value) -> ()
    ^bb5:  // pred: ^bb4
      %4 = "pdl_interp.get_operands"(%arg0) {index = 2 : i32} : (!pdl.operation) -> !pdl.range<value>
      "pdl_interp.is_not_null"(%4)[^bb6, ^bb9] : (!pdl.range<value>) -> ()
    ^bb6:  // pred: ^bb5
      %5 = "pdl_interp.get_operands"(%arg0) {index = 2 : i32} : (!pdl.operation) -> !pdl.value
      "pdl_interp.is_not_null"(%5)[^bb7, ^bb9] : (!pdl.value) -> ()
    ^bb7:  // pred: ^bb6
      %6 = "pdl_interp.get_operands"(%arg0) {index = 50 : i32} : (!pdl.operation) -> !pdl.value
      "pdl_interp.is_not_null"(%6)[^bb9, ^bb8] : (!pdl.value) -> ()
    ^bb8:  // pred: ^bb7
      "pdl_interp.record_match"(%arg0, %0, %2, %4, %5, %arg0)[^bb9] {benefit = 1 : i16, operand_segment_sizes = dense<[5, 1]> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>, !pdl.value, !pdl.operation) -> ()
    ^bb9:  // 9 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation, %arg1: !pdl.range<value>, %arg2: !pdl.range<value>, %arg3: !pdl.range<value>, %arg4: !pdl.value):
        %0 = "pdl_interp.create_operation"(%arg1) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (!pdl.range<value>) -> !pdl.operation
        %1 = "pdl_interp.create_operation"(%arg2) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (!pdl.range<value>) -> !pdl.operation
        %2 = "pdl_interp.create_operation"(%arg3) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (!pdl.range<value>) -> !pdl.operation
        %3 = "pdl_interp.create_operation"(%arg4) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (!pdl.value) -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>, !pdl.value) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0:5 = "test.producer"() : () -> (i32, i32, i32, i32, i32)
    "test.attr_sized_operands"(%0#0, %0#1, %0#2, %0#3, %0#4) {operand_segment_sizes = dense<[0, 4, 1, 0]> : vector<4 × i32>} : (i32, i32, i32, i32, i32) -> ()
  }) {sym_name = "ir", test.get_operands_2} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_result_count"(%arg0)[^bb1, ^bb3] {count = 5 : i32} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      %0 = "pdl_interp.get_result"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
      %1 = "pdl_interp.get_result"(%arg0) {index = 4 : i32} : (!pdl.operation) -> !pdl.value
      %2 = "pdl_interp.get_value_type"(%0) : (!pdl.value) -> !pdl.type
      %3 = "pdl_interp.get_value_type"(%1) : (!pdl.value) -> !pdl.type
      "pdl_interp.are_equal"(%2, %3)[^bb2, ^bb3] : (!pdl.type, !pdl.type) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0:5 = "test.op"() : () -> (i32, i32, i32, i32, i32)
    %1:5 = "test.op"() : () -> (i32, i32, i32, i32, i64)
  }) {sym_name = "ir", test.get_result_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_result_count"(%arg0)[^bb1, ^bb3] {count = 5 : i32} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      %0 = "pdl_interp.get_results"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.range<value>
      %1 = "pdl_interp.get_results"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      "pdl_interp.are_equal"(%0, %1)[^bb2, ^bb3] : (!pdl.range<value>, !pdl.range<value>) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0:5 = "test.producer"() : () -> (i32, i32, i32, i32, i32)
  }) {sym_name = "ir", test.get_results_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb9] {name = "test.attr_sized_results"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      %0 = "pdl_interp.get_results"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.range<value>
      "pdl_interp.is_not_null"(%0)[^bb2, ^bb9] : (!pdl.range<value>) -> ()
    ^bb2:  // pred: ^bb1
      %1 = "pdl_interp.get_results"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
      "pdl_interp.is_not_null"(%1)[^bb9, ^bb3] : (!pdl.value) -> ()
    ^bb3:  // pred: ^bb2
      %2 = "pdl_interp.get_results"(%arg0) {index = 1 : i32} : (!pdl.operation) -> !pdl.range<value>
      "pdl_interp.is_not_null"(%2)[^bb4, ^bb9] : (!pdl.range<value>) -> ()
    ^bb4:  // pred: ^bb3
      %3 = "pdl_interp.get_results"(%arg0) {index = 1 : i32} : (!pdl.operation) -> !pdl.value
      "pdl_interp.is_not_null"(%3)[^bb9, ^bb5] : (!pdl.value) -> ()
    ^bb5:  // pred: ^bb4
      %4 = "pdl_interp.get_results"(%arg0) {index = 2 : i32} : (!pdl.operation) -> !pdl.range<value>
      "pdl_interp.is_not_null"(%4)[^bb6, ^bb9] : (!pdl.range<value>) -> ()
    ^bb6:  // pred: ^bb5
      %5 = "pdl_interp.get_results"(%arg0) {index = 2 : i32} : (!pdl.operation) -> !pdl.value
      "pdl_interp.is_not_null"(%5)[^bb7, ^bb9] : (!pdl.value) -> ()
    ^bb7:  // pred: ^bb6
      %6 = "pdl_interp.get_results"(%arg0) {index = 50 : i32} : (!pdl.operation) -> !pdl.value
      "pdl_interp.is_not_null"(%6)[^bb9, ^bb8] : (!pdl.value) -> ()
    ^bb8:  // pred: ^bb7
      "pdl_interp.record_match"(%arg0, %0, %2, %4, %5, %arg0)[^bb9] {benefit = 1 : i16, operand_segment_sizes = dense<[5, 1]> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>, !pdl.value, !pdl.operation) -> ()
    ^bb9:  // 9 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation, %arg1: !pdl.range<value>, %arg2: !pdl.range<value>, %arg3: !pdl.range<value>, %arg4: !pdl.value):
        %0 = "pdl_interp.get_value_type"(%arg1) : (!pdl.range<value>) -> !pdl.range<type>
        %1 = "pdl_interp.get_value_type"(%arg2) : (!pdl.range<value>) -> !pdl.range<type>
        %2 = "pdl_interp.get_value_type"(%arg3) : (!pdl.range<value>) -> !pdl.range<type>
        %3 = "pdl_interp.get_value_type"(%arg4) : (!pdl.value) -> !pdl.type
        %4 = "pdl_interp.create_operation"(%0) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[0, 0, 1]> : vector<3 × i32>} : (!pdl.range<type>) -> !pdl.operation
        %5 = "pdl_interp.create_operation"(%1) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[0, 0, 1]> : vector<3 × i32>} : (!pdl.range<type>) -> !pdl.operation
        %6 = "pdl_interp.create_operation"(%2) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[0, 0, 1]> : vector<3 × i32>} : (!pdl.range<type>) -> !pdl.operation
        %7 = "pdl_interp.create_operation"(%3) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[0, 0, 1]> : vector<3 × i32>} : (!pdl.type) -> !pdl.operation
        %8 = "pdl_interp.get_results"(%4) : (!pdl.operation) -> !pdl.range<value>
        %9 = "pdl_interp.get_results"(%5) : (!pdl.operation) -> !pdl.range<value>
        %10 = "pdl_interp.get_results"(%6) : (!pdl.operation) -> !pdl.range<value>
        "pdl_interp.replace"(%arg0, %8, %9, %10) : (!pdl.operation, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>, !pdl.value) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0:5 = "test.attr_sized_results"() {result_segment_sizes = dense<[0, 4, 1, 0]> : vector<4 × i32>} : () -> (i32, i32, i32, i32, i32)
    "test.consumer"(%0#0, %0#1, %0#2, %0#3, %0#4) : (i32, i32, i32, i32, i32) -> ()
  }) {sym_name = "ir", test.get_results_2} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb3] {name = "test.op"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@failure} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 2 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 2 preds: ^bb0, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "failure"} : () -> ()
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() : () -> ()
  }) {sym_name = "ir", test.record_match_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb2] {name = "test.op"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      %0 = "pdl_interp.get_operands"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      %1 = "pdl_interp.get_results"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      %2 = "pdl_interp.get_value_type"(%1) : (!pdl.range<value>) -> !pdl.range<type>
      "pdl_interp.record_match"(%0, %2, %arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<[3, 1]> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.range<value>, !pdl.range<type>, !pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.range<value>, %arg1: !pdl.range<type>, %arg2: !pdl.operation):
        %0 = "pdl_interp.create_operation"(%arg0, %arg1) {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<[1, 0, 1]> : vector<3 × i32>} : (!pdl.range<value>, !pdl.range<type>) -> !pdl.operation
        %1 = "pdl_interp.get_results"(%0) : (!pdl.operation) -> !pdl.range<value>
        "pdl_interp.replace"(%arg2, %1) : (!pdl.operation, !pdl.range<value>) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.range<value>, !pdl.range<type>, !pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.producer"() : () -> i32
    %1:2 = "test.op"(%0) : (i32) -> (i64, i32)
    "test.consumer"(%1#0, %1#1) : (i64, i32) -> ()
  }) {sym_name = "ir", test.record_match_2} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.check_operation_name"(%arg0)[^bb1, ^bb2] {name = "test.op"} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.record_match"(%arg0, %arg0)[^bb2] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.get_operand"(%arg0) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
        "pdl_interp.replace"(%arg0, %0) : (!pdl.operation, !pdl.value) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op_input"() : () -> i32
    %1 = "test.op"(%0) : (i32) -> i32
    "test.op_consumer"(%1) : (i32) -> ()
  }) {sym_name = "ir", test.replace_op_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_attribute"(%arg0) {name = "test_attr"} : (!pdl.operation) -> !pdl.attribute
      "pdl_interp.switch_attribute"(%0)[^bb3, ^bb3, ^bb1] {caseValues = [0, unit]} : (!pdl.attribute) -> ()
    ^bb1:  // pred: ^bb0
      %1 = "pdl_interp.get_attribute"(%arg0) {name = "test_attr_2"} : (!pdl.operation) -> !pdl.attribute
      "pdl_interp.switch_attribute"(%1)[^bb2, ^bb3, ^bb3] {caseValues = [0, unit]} : (!pdl.attribute) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 5 preds: ^bb0, ^bb0, ^bb1, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() {test_attr} : () -> ()
  }) {sym_name = "ir", test.switch_attribute_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.switch_operand_count"(%arg0)[^bb3, ^bb3, ^bb1] {caseValues = dense<[0, 1]> : vector<2 × i32>} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.switch_operand_count"(%arg0)[^bb2, ^bb3, ^bb3] {caseValues = dense<[0, 2]> : vector<2 × i32>} : (!pdl.operation) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 5 preds: ^bb0, ^bb0, ^bb1, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op_input"() : () -> i32
    "test.op"(%0) : (i32) -> ()
  }) {sym_name = "ir", test.switch_operand_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.switch_operation_name"(%arg0)[^bb3, ^bb3, ^bb1] {caseValues = ["foo.op", "test.op"]} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.switch_operation_name"(%arg0)[^bb2, ^bb3, ^bb3] {caseValues = ["foo.op", "bar.op"]} : (!pdl.operation) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 5 preds: ^bb0, ^bb0, ^bb1, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() : () -> ()
  }) {sym_name = "ir", test.switch_operation_name_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      "pdl_interp.switch_result_count"(%arg0)[^bb3, ^bb3, ^bb1] {caseValues = dense<[0, 1]> : vector<2 × i32>} : (!pdl.operation) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.switch_result_count"(%arg0)[^bb2, ^bb3, ^bb3] {caseValues = dense<[0, 2]> : vector<2 × i32>} : (!pdl.operation) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 5 preds: ^bb0, ^bb0, ^bb1, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0 = "test.op"() : () -> i32
  }) {sym_name = "ir", test.switch_result_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_attribute"(%arg0) {name = "test_attr"} : (!pdl.operation) -> !pdl.attribute
      "pdl_interp.is_not_null"(%0)[^bb1, ^bb4] : (!pdl.attribute) -> ()
    ^bb1:  // pred: ^bb0
      %1 = "pdl_interp.get_attribute_type"(%0) : (!pdl.attribute) -> !pdl.type
      "pdl_interp.switch_type"(%1)[^bb4, ^bb2, ^bb4] {caseValues = [i32, i64]} : (!pdl.type) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.switch_type"(%1)[^bb3, ^bb4, ^bb4] {caseValues = [i16, i64]} : (!pdl.type) -> ()
    ^bb3:  // pred: ^bb2
      "pdl_interp.record_match"(%arg0, %arg0)[^bb4] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb4:  // 6 preds: ^bb0, ^bb1, ^bb1, ^bb2, ^bb2, ^bb3
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    "test.op"() {test_attr = 10 : i32} : () -> ()
  }) {sym_name = "ir", test.switch_type_1} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "pdl_interp.func"() ({
    ^bb0(%arg0: !pdl.operation):
      %0 = "pdl_interp.get_results"(%arg0) : (!pdl.operation) -> !pdl.range<value>
      %1 = "pdl_interp.get_value_type"(%0) : (!pdl.range<value>) -> !pdl.range<type>
      "pdl_interp.switch_types"(%1)[^bb3, ^bb1, ^bb3] {caseValues = [[i64, i64], [i32]]} : (!pdl.range<type>) -> ()
    ^bb1:  // pred: ^bb0
      "pdl_interp.switch_types"(%1)[^bb2, ^bb3, ^bb3] {caseValues = [[i32], [i64, i32]]} : (!pdl.range<type>) -> ()
    ^bb2:  // pred: ^bb1
      "pdl_interp.record_match"(%arg0, %arg0)[^bb3] {benefit = 1 : i16, operand_segment_sizes = dense<1> : vector<2 × i32>, rewriter = @rewriters::@success} : (!pdl.operation, !pdl.operation) -> ()
    ^bb3:  // 5 preds: ^bb0, ^bb0, ^bb1, ^bb1, ^bb2
      "pdl_interp.finalize"() : () -> ()
    }) {function_type = (!pdl.operation) -> (), sym_name = "matcher"} : () -> ()
    "builtin.module"() ({
      "pdl_interp.func"() ({
      ^bb0(%arg0: !pdl.operation):
        %0 = "pdl_interp.create_operation"() {inputAttributeNames = [], name = "test.success", operand_segment_sizes = dense<0> : vector<3 × i32>} : () -> !pdl.operation
        "pdl_interp.erase"(%arg0) : (!pdl.operation) -> ()
        "pdl_interp.finalize"() : () -> ()
      }) {function_type = (!pdl.operation) -> (), sym_name = "success"} : () -> ()
    }) {sym_name = "rewriters"} : () -> ()
  }) {sym_name = "patterns"} : () -> ()
  "builtin.module"() ({
    %0:2 = "test.op"() : () -> (i64, i64)
  }) {sym_name = "ir", test.switch_types_1} : () -> ()
}) : () -> ()

// -----
