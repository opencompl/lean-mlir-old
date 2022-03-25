"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    "scf.for"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: index):
      %0 = "fake_read"() : () -> index
      %1 = "fake_compute"(%0) : (index) -> index
      "fake_write"(%1) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index) -> (), sym_name = "hoist"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: f32):
    %0 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg4: index, %arg5: f32):
      %1 = "fake_read"() : () -> index
      %2 = "fake_compute"(%1) : (index) -> index
      "fake_write"(%2) : (index) -> ()
      "scf.yield"(%arg5) : (f32) -> ()
    }) : (index, index, index, f32) -> f32
    "func.return"(%0) : (f32) -> ()
  }) {function_type = (index, index, index, f32) -> f32, sym_name = "hoist2"} : () -> ()
}) : () -> ()

// -----
