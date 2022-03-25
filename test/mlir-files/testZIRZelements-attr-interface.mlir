"builtin.module"() ({
  %0 = "arith.constant"() {value = #test.i64_elements<[10, 11, 12, 13, 14] : tensor<5xi64>>} : () -> tensor<5xi64>
  %1 = "arith.constant"() {value = dense<[10, 11, 12, 13, 14]> : tensor<5xi64>} : () -> tensor<5xi64>
  %2 = "arith.constant"() {value = opaque<"_", "0xDEADBEEF"> : tensor<5xi64>} : () -> tensor<5xi64>
  %3 = "arith.constant"() {value = dense<> : tensor<0xi64>} : () -> tensor<0xi64>
}) : () -> ()

// -----
