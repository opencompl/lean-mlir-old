"builtin.module"() ({
^bb0:
}) : () -> ()


"builtin.module"() ({
^bb0:
}) {foo.attr = true} : () -> ()


"builtin.module"() ({
  %0 = "foo.result_op"() : () -> i32
}) : () -> ()


"builtin.module"() ({
^bb0:
}) : () -> ()


"builtin.module"() ({
^bb0:
}) : () -> ()


"builtin.module"() ({
  %0 = "op"() : () -> i32
}) : () -> ()


"builtin.module"() ({
  "builtin.module"() ({
    "builtin.module"() ({
    ^bb0:
    }) {foo.bar, sym_name = "bar"} : () -> ()
  }) : () -> ()
}) {sym_name = "foo"} : () -> ()


"builtin.module"() ({
^bb0:
}) {test.another_attribute = #dlti.dl_spec<>, test.random_attribute = #dlti.dl_spec<>} : () -> ()


