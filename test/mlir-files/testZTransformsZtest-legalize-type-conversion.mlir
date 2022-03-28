"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i16):
    "foo.return"(%arg0) : (i16) -> ()
  }) {function_type = (i16) -> (), sym_name = "test_invalid_arg_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i64):
    "foo.return"(%arg0) : (i64) -> ()
  }) {function_type = (i64) -> (), sym_name = "test_valid_arg_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.type_producer"() : () -> f16
    "foo.return"(%0) : (f16) -> ()
  }) {function_type = () -> (), sym_name = "test_invalid_result_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.type_producer"() : () -> f16
    "foo.return"(%0) : (f16) -> ()
  }) {function_type = () -> (), sym_name = "test_invalid_result_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.another_type_producer"() : () -> f32
    "foo.return"(%0) : (f32) -> ()
  }) {function_type = () -> (), sym_name = "test_transitive_use_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.another_type_producer"() : () -> f16
    "foo.return"(%0) : (f16) -> ()
  }) {function_type = () -> (), sym_name = "test_transitive_use_invalid_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.type_producer"() : () -> f32
    "foo.return"(%0) : (f32) -> ()
  }) {function_type = () -> (), sym_name = "test_valid_result_legalization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.signature_conversion_undo"() ({
    ^bb0(%arg0: f32):
      "test.type_consumer"(%arg0) : (f32) -> ()
      "test.return"(%arg0) : (f32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_signature_conversion_undo"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.unsupported_block_arg_type"() ({
    ^bb0(%arg0: index):
      "test.return"(%arg0) : (index) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_block_argument_not_converted"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.signature_conversion_no_converter"() ({
    ^bb0(%arg0: f32):
      "test.type_consumer"(%arg0) : (f32) -> ()
      "test.return"(%arg0) : (f32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_signature_conversion_no_converter"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.type_producer"() : () -> !test.test_rec<something, test_rec<something>>
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "recursive_type_conversion"} : () -> ()
}) : () -> ()


