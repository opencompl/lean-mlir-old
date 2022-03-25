"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "test.cast"(%arg0, %arg1) : (i32, i32) -> i32
    %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
    "test.return"(%0, %1) : (i32, i32) -> ()
  }) {function_type = (i32, i32) -> (), sym_name = "test1"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
^bb0:
}) : () -> ()

// -----
