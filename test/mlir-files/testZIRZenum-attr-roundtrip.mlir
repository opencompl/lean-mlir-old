"builtin.module"() ({
  "func.func"() ({
    "test.op"() {value = #test<"enum first">} : () -> ()
    "test.op"() {value = #test<"enum second">} : () -> ()
    "test.op"() {value = #test<"enum third">} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_enum_attr_roundtrip"} : () -> ()
  "func.func"() ({
    "test.op_with_enum"() {value = #test<"enum third">} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_op_with_enum"} : () -> ()
  "func.func"() ({
    "test.op_with_enum"() {tag = 0 : i32, value = #test<"enum third">} : () -> ()
    "test.op_with_enum"() {tag = 0 : i32, value = #test<"enum first">} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_match_op_with_enum"} : () -> ()
}) : () -> ()


