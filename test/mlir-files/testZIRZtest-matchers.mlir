"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = "arith.addf"(%arg0, %arg1) : (f32, f32) -> f32
    %1 = "arith.addf"(%arg0, %arg2) : (f32, f32) -> f32
    %2 = "arith.addf"(%arg2, %arg1) : (f32, f32) -> f32
    %3 = "arith.mulf"(%arg0, %2) : (f32, f32) -> f32
    %4 = "arith.mulf"(%3, %1) : (f32, f32) -> f32
    %5 = "arith.mulf"(%4, %4) : (f32, f32) -> f32
    %6 = "arith.mulf"(%5, %5) : (f32, f32) -> f32
    "func.return"() : () -> ()
  }) {function_type = (f32, f32, f32) -> (), sym_name = "test1"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %1 = "arith.addf"(%arg0, %0) : (f32, f32) -> f32
    %2 = "arith.mulf"(%arg0, %1) : (f32, f32) -> f32
    "func.return"(%2) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "test2"} : () -> ()
}) : () -> ()


