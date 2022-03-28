"builtin.module"() ({
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dead_function", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dead_function_b"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @dead_function_b} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "live_function"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "live_function_b"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @live_function_b} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dead_function_c", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @dead_function_c} : () -> ()
    "func.call"() {callee = @dead_function_c} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dead_function_d", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @dead_function_c} : () -> ()
    "func.call"() {callee = @dead_function_d} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "live_function_c"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "live_function_d", sym_visibility = "private"} : () -> ()
  "live.user"() {use = @live_function_d} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "func.call"() {callee = @dead_function_e} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "live_function_e"} : () -> ()
  "func.func"() ({
    "test.fold_to_call_op"() {callee = @dead_function_f} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dead_function_e", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dead_function_f", sym_visibility = "private"} : () -> ()
}) : () -> ()


