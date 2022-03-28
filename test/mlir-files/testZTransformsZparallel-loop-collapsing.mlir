"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 2 : index} : () -> index
    %3 = "arith.constant"() {value = 3 : index} : () -> index
    %4 = "arith.constant"() {value = 4 : index} : () -> index
    %5 = "arith.constant"() {value = 5 : index} : () -> index
    %6 = "arith.constant"() {value = 6 : index} : () -> index
    %7 = "arith.constant"() {value = 7 : index} : () -> index
    %8 = "arith.constant"() {value = 8 : index} : () -> index
    %9 = "arith.constant"() {value = 9 : index} : () -> index
    %10 = "arith.constant"() {value = 10 : index} : () -> index
    %11 = "arith.constant"() {value = 11 : index} : () -> index
    %12 = "arith.constant"() {value = 12 : index} : () -> index
    %13 = "arith.constant"() {value = 13 : index} : () -> index
    %14 = "arith.constant"() {value = 14 : index} : () -> index
    %15 = "arith.constant"() {value = 15 : index} : () -> index
    %16 = "arith.constant"() {value = 26 : index} : () -> index
    "scf.parallel"(%0, %3, %6, %9, %12, %2, %5, %8, %16, %14, %1, %4, %7, %10, %13) ({
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
      %17 = "magic.op"(%arg0, %arg1, %arg2, %arg3, %arg4) : (index, index, index, index, index) -> index
      "scf.yield"() : () -> ()
    }) {operand_segment_sizes = dense<[5, 5, 5, 0]> : vector<4 × i32>} : (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "parallel_many_dims"} : () -> ()
}) : () -> ()


