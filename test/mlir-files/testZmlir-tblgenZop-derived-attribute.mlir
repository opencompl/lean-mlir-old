"builtin.module"() ({
  "func.func"() ({
    %0 = "test.derived_type_attr"() : () -> tensor<10xf32>
    %1 = "test.derived_type_attr"() : () -> tensor<12xi79>
    %2 = "test.derived_type_attr"() : () -> tensor<12xcomplex<f32>>
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "verifyDerivedAttributes"} : () -> ()
}) : () -> ()

// -----
