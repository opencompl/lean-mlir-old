"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !test.smpla):
    "func.return"() : () -> ()
  }) {function_type = (!test.smpla) -> (), sym_name = "simpleA"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !test.cmpnd_a<1, !test.smpla, [5, 6]>):
    "func.return"() : () -> ()
  }) {function_type = (!test.cmpnd_a<1, !test.smpla, [5, 6]>) -> (), sym_name = "compoundA"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>):
    "func.return"() : () -> ()
  }) {function_type = (!test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>) -> (), sym_name = "compoundNested"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>):
    "test.format_cpmd_nested_type"(%arg0) : (!test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>) -> ()
    "test.format_cpmd_nested_type"(%arg0) : (!test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (!test.cmpnd_nested_outer<i <42 <1, !test.smpla, [5, 6]>>>) -> (), sym_name = "compoundNestedExplicit"} : () -> ()
  "func.func"() ({
  }) {function_type = (!test.cmpnd_nested_outer_qual<i !test.cmpnd_inner<42 <1, !test.smpla, [5, 6]>>>) -> (), sym_name = "compoundNestedQual", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !test.int<signed, 8>, %arg1: !test.int<unsigned, 2>, %arg2: !test.int<none, 1>):
    "func.return"() : () -> ()
  }) {function_type = (!test.int<signed, 8>, !test.int<unsigned, 2>, !test.int<none, 1>) -> (), sym_name = "testInt"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !test.struct<{field1,!test.smpla}, {field2,!test.int<none, 3>}>):
    "func.return"() : () -> ()
  }) {function_type = (!test.struct<{field1,!test.smpla}, {field2,!test.int<none, 3>}>) -> (), sym_name = "structTest"} : () -> ()
}) : () -> ()

// -----
