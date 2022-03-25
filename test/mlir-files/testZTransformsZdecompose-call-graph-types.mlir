"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tuple<i1, i32>):
    "func.return"(%arg0) : (tuple<i1, i32>) -> ()
  }) {function_type = (tuple<i1, i32>) -> tuple<i1, i32>, sym_name = "identity"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tuple<i1>):
    "func.return"(%arg0) : (tuple<i1>) -> ()
  }) {function_type = (tuple<i1>) -> tuple<i1>, sym_name = "identity_1_to_1_no_materializations"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tuple<tuple<tuple<i1>>>):
    "func.return"(%arg0) : (tuple<tuple<tuple<i1>>>) -> ()
  }) {function_type = (tuple<tuple<tuple<i1>>>) -> tuple<tuple<tuple<i1>>>, sym_name = "recursive_decomposition"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (tuple<i1, i32>) -> tuple<i1, i32>, sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tuple<i1, i32>):
    %0 = "func.call"(%arg0) {callee = @callee} : (tuple<i1, i32>) -> tuple<i1, i32>
    "func.return"(%0) : (tuple<i1, i32>) -> ()
  }) {function_type = (tuple<i1, i32>) -> tuple<i1, i32>, sym_name = "caller"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (tuple<>) -> tuple<>, sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tuple<>):
    %0 = "func.call"(%arg0) {callee = @callee} : (tuple<>) -> tuple<>
    "func.return"(%0) : (tuple<>) -> ()
  }) {function_type = (tuple<>) -> tuple<>, sym_name = "caller"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    %0 = "test.source"() : () -> tuple<i1, i32>
    "func.return"(%0) : (tuple<i1, i32>) -> ()
  }) {function_type = () -> tuple<i1, i32>, sym_name = "unconverted_op_result"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (tuple<>, i1, tuple<i2>, i3, tuple<i4, i5>, i6) -> (tuple<>, i1, tuple<i2>, i3, tuple<i4, i5>, i6), sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tuple<>, %arg1: i1, %arg2: tuple<i2>, %arg3: i3, %arg4: tuple<i4, i5>, %arg5: i6):
    %0:6 = "func.call"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {callee = @callee} : (tuple<>, i1, tuple<i2>, i3, tuple<i4, i5>, i6) -> (tuple<>, i1, tuple<i2>, i3, tuple<i4, i5>, i6)
    "func.return"(%0#0, %0#1, %0#2, %0#3, %0#4, %0#5) : (tuple<>, i1, tuple<i2>, i3, tuple<i4, i5>, i6) -> ()
  }) {function_type = (tuple<>, i1, tuple<i2>, i3, tuple<i4, i5>, i6) -> (tuple<>, i1, tuple<i2>, i3, tuple<i4, i5>, i6), sym_name = "caller"} : () -> ()
}) : () -> ()

// -----
