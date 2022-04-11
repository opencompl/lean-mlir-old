





!test_tuple = type tuple<!test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla>
!test_ui8_ = type !test.int<unsigned, 8>
!tuple = type tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>
"builtin.module"() ({
  "test.op"() {alias_test = #test2Ealias} : () -> ()
  "test.op"() {alias_test = #test_alias0_} : () -> ()
  "test.op"() {alias_test = #_0_test_alias} : () -> ()
  "test.op"() {alias_test = [#test_alias_conflict0_0, #test_alias_conflict0_1]} : () -> ()
  %0 = "test.op"() {alias_test = "alias_test:large_tuple"} : () -> !tuple
  %1 = "test.op"() {alias_test = "alias_test:large_tuple"} : () -> !test_tuple
  %2 = "test.op"() : () -> tensor<32 × f32, #test_encoding>
  %3 = "test.op"() : () -> tensor<32 × !test_ui8_>
}) : () -> ()


