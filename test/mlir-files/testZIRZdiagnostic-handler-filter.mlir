#loc = loc(callsite("foo"("mysource1":0:0) at callsite("mysource2":1:0 at "mysource3":2:0)))
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "test1", sym_visibility = "private", test.loc = #loc} : () -> ()
}) : () -> ()


#loc = loc("mysource1":0:0)
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "test2", sym_visibility = "private", test.loc = #loc} : () -> ()
}) : () -> ()


