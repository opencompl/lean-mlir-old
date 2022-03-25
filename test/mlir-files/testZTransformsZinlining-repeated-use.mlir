"builtin.module"() ({
  "func.func"() ({
    "func.call"() {callee = @B, inA} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "A"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @E} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "B"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @D} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "C"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @B, inD} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "D", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @fabsf} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "E"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "fabsf", sym_visibility = "private"} : () -> ()
}) : () -> ()

// -----
