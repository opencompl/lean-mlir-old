"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × f32>, %arg1: tensor<4 × f32>):
    %0 = "arith.addf"(%arg0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %1 = "arith.addf"(%arg0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %2 = "arith.addf"(%arg0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %3 = "arith.addf"(%arg0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %4 = "arith.addf"(%arg0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %5 = "arith.addf"(%arg0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %6 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %7 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %8 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %9 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %10 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %11 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %12 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %13 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %14 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %15 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %16 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %17 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %18 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %19 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %20 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %21 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %22 = "xla.add"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    %23 = "long_op_name"(%0, %arg1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    "func.return"(%1) : (tensor<4 × f32>) -> ()
  }) {function_type = (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>, sym_name = "main"} : () -> ()
}) : () -> ()


