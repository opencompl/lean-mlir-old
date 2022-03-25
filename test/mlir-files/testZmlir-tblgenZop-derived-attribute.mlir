"builtin.module"() ({
  "func.func"() ({
    %0 = "test.derived_type_attr"() : () -> tensor<10 × f32>
    %1 = "test.derived_type_attr"() : () -> tensor<12 × i79>
    %2 = "test.derived_type_attr"() : () -> tensor<12 × comple × <f32>>
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "verifyDerivedAttributes"} : () -> ()
}) : () -> ()

// -----
