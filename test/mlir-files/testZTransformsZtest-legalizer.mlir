"builtin.module"() ({
  "func.func"() ({
    %0 = "test.illegal_op_a"() : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "verifyDirectPattern"} : () -> ()
  "func.func"() ({
    %0 = "test.illegal_op_c"() : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "verifyLargerBenefit"} : () -> ()
  "func.func"() ({
  }) {function_type = (i16) -> (), sym_name = "remap_input_1_to_0", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i64):
    "test.invalid"(%arg0) : (i64) -> ()
  }) {function_type = (i64) -> (), sym_name = "remap_input_1_to_1"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i64):
    "func.call"(%arg0) {callee = @remap_input_1_to_1} : (i64) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i64) -> (), sym_name = "remap_call_1_to_1"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: f32):
    "test.return"(%arg0) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "remap_input_1_to_N"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: f32):
    "work"(%arg0) : (f32) -> ()
  }) {function_type = (f32) -> (), sym_name = "remap_input_1_to_N_remaining_use"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i42):
    "test.return"(%arg0) : (i42) -> ()
  }) {function_type = (i42) -> (), sym_name = "remap_materialize_1_to_1"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    "work"(%arg0) : (index) -> ()
  }) {function_type = (index) -> (), sym_name = "remap_input_to_self"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i64, %arg1: i16, %arg2: i64):
    "test.invalid"(%arg0, %arg2) : (i64, i64) -> ()
  }) {function_type = (i64, i16, i64) -> (i64, i64), sym_name = "remap_multi"} : () -> ()
  "func.func"() ({
    "foo.region"() ({
    ^bb0(%arg0: i64, %arg1: i16, %arg2: i64):
      "test.invalid"(%arg0, %arg2) : (i64, i64) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "no_remap_nested"} : () -> ()
  "func.func"() ({
    "test.region"() ({
    ^bb0(%arg0: i64, %arg1: i16, %arg2: i64, %arg3: f32):
      "test.invalid"(%arg0, %arg2, %arg3) : (i64, i64, f32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "remap_moved_region_args"} : () -> ()
  "func.func"() ({
    "test.region"() ({
    ^bb0(%arg0: i64, %arg1: i16, %arg2: i64, %arg3: f32):
      "test.invalid"(%arg0, %arg2, %arg3) : (i64, i64, f32) -> ()
    }) {legalizer.should_clone} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "remap_cloned_region_args"} : () -> ()
  "func.func"() ({
    "test.drop_region_op"() ({
    ^bb0(%arg0: i64, %arg1: i16, %arg2: i64, %arg3: f32):
      "test.invalid"(%arg0, %arg2, %arg3) : (i64, i64, f32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "remap_drop_region"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i16, %arg1: i64):
    "work"(%arg0) : (i16) -> ()
  }) {function_type = (i16, i64) -> (), sym_name = "dropped_input_in_use"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i8):
    %0 = "test.rewrite"(%arg0) : (i8) -> i8
    %1 = "test.rewrite"(%0) : (i8) -> i8
    "func.return"(%1) : (i8) -> ()
  }) {function_type = (i8) -> i8, sym_name = "up_to_date_replacement"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.op_with_region_fold"(%arg0) ({
      "foo.op_with_region_terminator"() : () -> ()
    }) : (i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "remove_foldable_op"} : () -> ()
  "func.func"() ({
    "test.create_block"() : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "create_block"} : () -> ()
  "func.func"() ({
    "test.recursive_rewrite"() {depth = 3 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "bounded_recursion"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.illegal_op_f"() : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "fail_to_convert_illegal_op"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.region_builder"() : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "fail_to_convert_illegal_op_in_region"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.region"() ({
    ^bb0(%arg0: i64):
      "test.region_builder"() : () -> ()
      "test.valid"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "fail_to_convert_region"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.create_illegal_block"() : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "create_illegal_block"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.undo_block_arg_replace"() ({
    ^bb0(%arg0: i32):
      "test.return"(%arg0) : (i32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "undo_block_arg_replace"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.undo_block_erase"() ({
      "unregistered.return"()[^bb1] : () -> ()
    ^bb1:  
      "unregistered.return"() : () -> ()
    }) : () -> ()
  }) {function_type = () -> (), sym_name = "undo_block_erase"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.illegal_op_with_region_anchor"() : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "undo_child_created_before_parent"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.blackhole_producer"() : () -> i32
    "test.blackhole"(%0) : (i32) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "blackhole"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.illegal_op_g"() : () -> i32
    "test.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "create_unregistered_op_in_pattern"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "test.passthrough_fold"(%arg0) : (f32) -> i32
    "test.return"(%0) : (i32) -> ()
  }) {function_type = (f32) -> i32, sym_name = "typemismatch"} : () -> ()
}) : () -> ()


