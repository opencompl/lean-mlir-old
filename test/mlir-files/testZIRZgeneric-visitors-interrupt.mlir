"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() {interrupt_before_all = true} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() ({
      "bar"() : () -> ()
    }) {interrupt_after_all = true} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() ({
      "bar0"() : () -> ()
    }, {
      "bar1"() : () -> ()
    }) {interrupt_after_region = 0 : i64} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    "foo"() : () -> ()
    "test.two_region_op"() ({
      "work"() : () -> ()
    }, {
      "work"() : () -> ()
    }) {interrupt_after_all = true} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    "foo"() : () -> ()
    "test.two_region_op"() ({
      "work"() : () -> ()
    }, {
      "work"() : () -> ()
    }) {interrupt_after_region = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() ({
      "bar0"() : () -> ()
    }, {
      "bar1"() : () -> ()
    }) {skip_before_all = true} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() ({
      "bar0"() : () -> ()
    }, {
      "bar1"() : () -> ()
    }) {skip_after_all = true} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() ({
      "bar0"() : () -> ()
    }, {
      "bar1"() : () -> ()
    }) {skip_after_region = 0 : i64} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()

// -----
