"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 3 : index} : () -> index
    %1 = "arith.constant"() {value = 7 : index} : () -> index
    %2 = "arith.constant"() {value = 11 : index} : () -> index
    %3 = "arith.constant"() {value = 29 : index} : () -> index
    %4 = "arith.constant"() {value = 3 : index} : () -> index
    %5 = "arith.constant"() {value = 4 : index} : () -> index
    "scf.parallel"(%0, %1, %2, %3, %4, %5) ({
    ^bb0(%arg0: index, %arg1: index):
      %6 = "magic.op"(%arg0, %arg1) : (index, index) -> index
      "scf.yield"() : () -> ()
    }) {operand_segment_sizes = dense<[2, 2, 2, 0]> : vector<4 × i32>} : (index, index, index, index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "collapse_to_single"} : () -> ()
}) : () -> ()

// -----
