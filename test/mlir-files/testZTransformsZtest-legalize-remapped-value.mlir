"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "test.one_variadic_out_one_variadic_in1"(%arg0) : (i32) -> i32
    %1 = "test.one_variadic_out_one_variadic_in1"(%0) : (i32) -> i32
    "test.return"() : () -> ()
  }) {function_type = (i32) -> (), sym_name = "remap_input_1_to_1"} : () -> ()
  "func.func"() ({
    %0 = "test.remapped_value_region"() ({
      %1 = "test.type_producer"() : () -> f32
      "test.return"(%1) : (f32) -> ()
    }) : () -> f32
    "test.type_consumer"(%0) : (f32) -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "remap_unconverted"} : () -> ()
}) : () -> ()

// -----
