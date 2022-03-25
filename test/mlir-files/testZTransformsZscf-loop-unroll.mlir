"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 10 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "scf.for"(%0, %1, %2, %arg0) ({
    ^bb0(%arg2: index, %arg3: f32):
      %4 = "arith.addf"(%arg3, %arg1) : (f32, f32) -> f32
      "scf.yield"(%4) : (f32) -> ()
    }) : (index, index, index, f32) -> f32
    "func.return"(%3) : (f32) -> ()
  }) {function_type = (f32, f32) -> f32, sym_name = "scf_loop_unroll_single"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: index):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2:2 = "scf.for"(%0, %arg2, %1, %arg0, %arg1) ({
    ^bb0(%arg3: index, %arg4: f32, %arg5: f32):
      %3 = "arith.addf"(%arg4, %arg0) : (f32, f32) -> f32
      %4 = "arith.addf"(%arg5, %arg1) : (f32, f32) -> f32
      "scf.yield"(%3, %4) : (f32, f32) -> ()
    }) : (index, index, index, f32, f32) -> (f32, f32)
    "func.return"(%2#0, %2#1) : (f32, f32) -> ()
  }) {function_type = (f32, f32, index) -> (f32, f32), sym_name = "scf_loop_unroll_double_symbolic_ub"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    "scf.for"(%1, %2, %0) ({
    ^bb0(%arg0: index):
      %3 = "test.foo"(%arg0) : (index) -> i32
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "scf_loop_unroll_factor_1_promote"} : () -> ()
}) : () -> ()

// -----
