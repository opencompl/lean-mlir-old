"builtin.module"() ({
  "func.func"() ({
    %0 = "test.type_producer"() : () -> i32
    "test.type_consumer"(%0) : (i32) -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "multi_level_mapping"} : () -> ()
  "func.func"() ({
    "test.drop_region_op"() ({
      %0 = "test.illegal_op_f"() : () -> i32
      "test.return"() : () -> ()
    }) : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dropped_region_with_illegal_ops"} : () -> ()
  "func.func"() ({
    %0 = "test.replace_non_root"() : () -> i32
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "replace_non_root_illegal_op"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    "builtin.module"() ({
      %0 = "test.illegal_op_f"() : () -> i32
    }) {test.recursively_legal} : () -> ()
    "func.func"() ({
    ^bb0(%arg0: i64):
      %0 = "test.illegal_op_f"() : () -> i32
      "test.return"() : () -> ()
    }) {function_type = (i64) -> (), sym_name = "dynamic_func", test.recursively_legal} : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "recursively_legal_invalid_op"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    "test.region"() ({
    ^bb0(%arg0: i64):
      "test.invalid"(%arg0) : (i64) -> ()
    }) {legalizer.should_clone} : () -> ()
    %0 = "test.illegal_op_f"() : () -> i32
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_undo_region_clone"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    "foo.unknown_op"() {test.dynamically_legal} : () -> ()
    "foo.unknown_op"() : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_unknown_dynamically_legal"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    "test.region"() ({
    ^bb0(%arg0: i64):
      "cf.br"(%arg0)[^bb1] : (i64) -> ()
    ^bb1(%0: i64):  // pred: ^bb0
      "test.invalid"(%0) : (i64) -> ()
    }) : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_undo_region_inline"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    "test.region"() ({
    ^bb0(%arg0: i64):
      "cf.br"(%arg0)[^bb1] : (i64) -> ()
    ^bb1(%0: i64):  // pred: ^bb0
      "test.invalid"(%0) : (i64) -> ()
    }) {legalizer.erase_old_blocks, legalizer.should_clone} : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_undo_block_erase"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    %0 = "test.illegal_op_g"() : () -> i32
    "test.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "create_unregistered_op_in_pattern"} : () -> ()
}) : () -> ()

// -----
