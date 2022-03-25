"builtin.module"() ({
  "func.func"() ({
    %0 = "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<a, test_rec<b, test_type>>
    %1 = "test.dummy_op_for_roundtrip"() : () -> !test.test_rec<c, test_rec<c>>
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "roundtrip"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "create"} : () -> ()
}) : () -> ()

// -----
