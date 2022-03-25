"builtin.module"() ({
  "test.top_level_op"() : () -> ()
  "test.top_level_op"() : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.top_level_op_strict_loc"() {strict_loc_check} : () -> ()
  "test.top_level_op_strict_loc"() {strict_loc_check} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.top_level_op_loc_match"() {strict_loc_check} : () -> ()
  "test.top_level_op_loc_match"() {strict_loc_check} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.top_level_op_block_loc_mismatch"() ({
  ^bb0(%arg0: i32):
  }) {strict_loc_check} : () -> ()
  "test.top_level_op_block_loc_mismatch"() ({
  ^bb0(%arg0: i32):
  }) {strict_loc_check} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.top_level_op_block_loc_match"() ({
  ^bb0(%arg0: i32):
  }) {strict_loc_check} : () -> ()
  "test.top_level_op_block_loc_match"() ({
  ^bb0(%arg0: i32):
  }) {strict_loc_check} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.top_level_name_mismatch"() : () -> ()
  "test.top_level_name_mismatch2"() : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.top_level_op_attr_mismatch"() {foo = "bar"} : () -> ()
  "test.top_level_op_attr_mismatch"() {foo = "bar2"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.top_level_op_cfg"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0)[^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%0: f32):  // pred: ^bb0
    "test.some_branching_op"() : () -> ()
  ^bb2(%1: i32):  // pred: ^bb0
    "test.some_branching_op"() : () -> ()
  }, {
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0)[^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%0: f32):  // pred: ^bb0
    "test.some_branching_op"() : () -> ()
  ^bb2(%1: i32):  // pred: ^bb0
    "test.some_branching_op"() : () -> ()
  }) {attr = "foo"} : () -> ()
  "test.top_level_op_cfg"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0)[^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%0: f32):  // pred: ^bb0
    "test.some_branching_op"() : () -> ()
  ^bb2(%1: i32):  // pred: ^bb0
    "test.some_branching_op"() : () -> ()
  }, {
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0)[^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%0: f32):  // pred: ^bb0
    "test.some_branching_op"() : () -> ()
  ^bb2(%1: i32):  // pred: ^bb0
    "test.some_branching_op"() : () -> ()
  }) {attr = "foo"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.operand_num_mismatch"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0) : (f32, i32) -> ()
  }) : () -> ()
  "test.operand_num_mismatch"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1) : (f32) -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.operand_type_mismatch"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0) : (f32, i32) -> ()
  }) : () -> ()
  "test.operand_type_mismatch"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg1) : (f32, f32) -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.block_type_mismatch"() ({
  ^bb0(%arg0: f32, %arg1: f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()
  "test.block_type_mismatch"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.block_arg_num_mismatch"() ({
  ^bb0(%arg0: f32, %arg1: f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()
  "test.block_arg_num_mismatch"() ({
  ^bb0(%arg0: f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.dataflow_match"() ({
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()
  "test.dataflow_match"() ({
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "test.dataflow_mismatch"() ({
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()
  "test.dataflow_mismatch"() ({
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.consumer"(%0#1, %0#0) : (i32, i32) -> ()
  }) : () -> ()
}) : () -> ()

// -----
