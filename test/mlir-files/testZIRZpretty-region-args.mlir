"builtin.module"() ({
  "func.func"() ({
    "test.polyfor"() ({
    ^bb0(%arg0: index, %arg1: index, %arg2: index):
      "foo"() : () -> ()
    }) {arg_names = ["i", "j", "k"]} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "custom_region_names"} : () -> ()
  "func.func"() ({
    "test.polyfor"() ({
    ^bb0(%arg0: i32, %arg1: i32, %arg2: index):
      %0 = "foo"() : () -> i32
    }) {arg_names = ["a .^x", "0"]} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "weird_names"} : () -> ()
}) : () -> ()


