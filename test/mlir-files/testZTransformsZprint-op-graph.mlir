"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "arith.constant"() {value = dense<[[0, 1], [2, 3]]> : tensor<2 × 2 × i32>} : () -> tensor<2 × 2 × i32>
    %1 = "arith.constant"() {value = dense<1> : tensor<5 × i32>} : () -> tensor<5 × i32>
    %2 = "arith.constant"() {value = dense<[[0, 1]]> : tensor<1 × 2 × i32>} : () -> tensor<1 × 2 × i32>
    %3 = "arith.constant"() {value = 10 : i32} : () -> i32
    %4 = "test.func"() : () -> i32
    %5:2 = "test.merge_blocks"() ({
      "test.br"(%arg0, %4, %3)[^bb1] : (i32, i32, i32) -> ()
    ^bb1(%6: i32, %7: i32, %8: i32):  
      "test.return"(%6, %7) : (i32, i32) -> ()
    }) : () -> (i32, i32)
    "test.return"(%5#0, %5#1) : (i32, i32) -> ()
  }) {function_type = (i32, i32) -> (), sym_name = "merge_blocks"} : () -> ()
}) : () -> ()


