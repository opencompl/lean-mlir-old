"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.functional_region_op"() ({
    ^bb0(%arg1: i32):
      %2 = "arith.addi"(%arg1, %arg1) : (i32, i32) -> i32
      "test.return"(%2) : (i32) -> ()
    }) : () -> ((i32) -> i32)
    %1 = "func.call_indirect"(%0, %arg0) : ((i32) -> i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "inline_with_arg"} : () -> ()
  "func.func"() ({
    %0 = "test.functional_region_op"() ({
      "test.region"() ({
        "foo.noinline_operation"() : () -> ()
      }) : () -> ()
      "test.return"() : () -> ()
    }) : () -> (() -> ())
    "func.call_indirect"(%0) : (() -> ()) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "no_inline_invalid_nested_operation"} : () -> ()
  "func.func"() ({
    %0 = "test.functional_region_op"() ({
      %1 = "test.functional_region_op"() ({
        "foo.noinline_operation"() : () -> ()
      }) : () -> (() -> ())
      "test.return"() : () -> ()
    }) : () -> (() -> ())
    "func.call_indirect"(%0) : (() -> ()) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "inline_ignore_invalid_nested_operation"} : () -> ()
  "func.func"() ({
    "foo.unknown_region"() ({
      %0 = "test.functional_region_op"() ({
        "test.return"() : () -> ()
      }) : () -> (() -> ())
      "func.call_indirect"(%0) : (() -> ()) -> ()
      "test.return"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "no_inline_invalid_dest_region"} : () -> ()
}) : () -> ()


