"builtin.module"() ({
  "func.func"() ({
  }) {foo = #test.cmpnd_a<1, !test.smpla, [5, 6]>, function_type = () -> (), sym_name = "compoundA", sym_visibility = "private"} : () -> ()
  %0 = "test.result_has_same_type_as_attr"() {attr = #test<"attr_with_self_type_param i32">} : () -> i32
  %1 = "test.result_has_same_type_as_attr"() {attr = #test<"attr_with_type_builder 10 : i16">} : () -> i16
  "func.func"() ({
  }) {foo = #test.cmpnd_nested_outer_qual<i #test.cmpnd_nested_inner<42 <1, !test.smpla, [5, 6]>>>, function_type = () -> (), sym_name = "qualifiedAttr", sym_visibility = "private"} : () -> ()
}) : () -> ()


