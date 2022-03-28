"builtin.module"() ({
  %0:4 = "dialect.op1"() : () -> (i1, i16, i32, i64)
  "dialect.op2"(%0#0, %0#2) : (i1, i32) -> ()
  "dialect.op3"() ({
  ^bb0(%arg0: i1):
    "dialect.innerop1"(%arg0, %0#2) : (i1, i32) -> ()
  }) : () -> ()
}) : () -> ()


