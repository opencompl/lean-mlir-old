"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    %0:3 = "test.wrapping_region"() ({
      %1:3 = "some.op"(%arg1, %arg0) {test.attr = "attr"} : (f32, i32) -> (i1, i2, i3)
      "test.return"(%1#0, %1#1, %1#2) : (i1, i2, i3) -> ()
    }) : () -> (i1, i2, i3)
    "func.return"(%0#2, %0#1, %0#0) : (i3, i2, i1) -> ()
  }) {function_type = (i32, f32) -> (i3, i2, i1), sym_name = "wrapping_op"} : () -> ()
}) : () -> ()

// -----
