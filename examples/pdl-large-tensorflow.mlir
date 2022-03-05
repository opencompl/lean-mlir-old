// RUN: mlir-opt %s  -allow-unregistered-dialect -test-pdl-bytecode-pass -split-input-file | FileCheck %s

// -----

//===----------------------------------------------------------------------===//
// 1-layer perceptron with split fwd/bwd operations
//===----------------------------------------------------------------------===//

module @patterns {
  // fc_fwd
  pdl.pattern : benefit(1) {
    %in_type = pdl.type
    %out_type = pdl.type
    %weight_type = pdl.type
    %rxact = pdl.operand : %in_type
    %weight = pdl.operand : %weight_type

    %attr0 = pdl.attribute false
    %op0 = pdl.operation "tf.MatMul" (%rxact, %weight : !pdl.value, !pdl.value) {"transpose_a" = %attr0, "transpose_b" = %attr0} -> (%out_type : !pdl.type)

    pdl.rewrite %op0 {
      %op1 = pdl.operation "kernel.FcFwd" (%rxact, %weight : !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
      %val1 = pdl.result 0 of %op1  // txact
      pdl.replace %op0 with (%val1 : !pdl.value)  // tf.MatMul
    }
  }

  // fc_bwd
  pdl.pattern : benefit(4) {
    %in_type = pdl.type
    %out_type = pdl.type
    %weight_type = pdl.type
    %const_type = pdl.type
    %rxact = pdl.operand : %in_type
    %rxdelta = pdl.operand : %out_type
    %weight = pdl.operand : %weight_type

    %attr0 = pdl.attribute true
    %attr1 = pdl.attribute false
    %op0 = pdl.operation "tf.MatMul" (%rxact, %rxdelta : !pdl.value, !pdl.value) {"transpose_a" = %attr0, "transpose_b" = %attr1} -> (%weight_type : !pdl.type)
    %val0 = pdl.result 0 of %op0
    %op1 = pdl.operation "tf.Const" -> (%const_type : !pdl.type)
    %val1 = pdl.result 0 of %op1
    %op2 = pdl.operation "tf.Mul" (%val0, %val1 : !pdl.value, !pdl.value) -> (%weight_type : !pdl.type)
    %val2 = pdl.result 0 of %op2
    %op3 = pdl.operation "tf.Sub" (%weight, %val2 : !pdl.value, !pdl.value) -> (%weight_type : !pdl.type)

    pdl.rewrite %op3 {
      %op4 = pdl.operation "kernel.FcBwd" (%rxact, %rxdelta, %weight : !pdl.value, !pdl.value, !pdl.value) -> (%weight_type : !pdl.type)
      %val4 = pdl.result 0 of %op4  // weight_out
      pdl.replace %op3 with (%val4 : !pdl.value)  // tf.Sub
      pdl.erase %op2  // tf.Mul
      pdl.erase %op1  // tf.Const
      pdl.erase %op0  // tf.MatMul
    }
  }

  // softmax_cross_entropy
  pdl.pattern : benefit(6) {
    %in_type = pdl.type
    %label_type = pdl.type
    %loss_type = pdl.type
    %mean_loss_type = pdl.type
    %mean_const_type = pdl.type
    %mul_const_type = pdl.type
    %rxact = pdl.operand : %in_type
    %rxlabel = pdl.operand : %label_type

    %op0 = pdl.operation "tf.SparseSoftmaxCrossEntropyWithLogits" (%rxact, %rxlabel : !pdl.value, !pdl.value) -> (%loss_type, %in_type : !pdl.type, !pdl.type)
    %val0_0 = pdl.result 0 of %op0  // loss
    %val0_1 = pdl.result 1 of %op0  // gradient
    %op1 = pdl.operation "tf.Const" -> (%mean_const_type : !pdl.type)
    %val1 = pdl.result 0 of %op1
    %op2 = pdl.operation "tf.Mean" (%val0_0, %val1 : !pdl.value, !pdl.value) -> (%mean_loss_type : !pdl.type)
    %val2 = pdl.result 0 of %op2
    %op3 = pdl.operation "tf.PreventGradient" (%val0_1 : !pdl.value) -> (%in_type : !pdl.type)
    %val3 = pdl.result 0 of %op3
    %op4 = pdl.operation "tf.Const" -> (%mul_const_type : !pdl.type)
    %val4 = pdl.result 0 of %op4
    %op5 = pdl.operation "tf.Mul" (%val3, %val4 : !pdl.value, !pdl.value) -> (%in_type : !pdl.type)

    pdl.rewrite {  // roots: %op2, %op5
      %op6 = pdl.operation "kernel.SoftmaxCrossEntropy" (%rxact, %rxlabel : !pdl.value, !pdl.value) -> (%mean_loss_type, %in_type : !pdl.type, !pdl.type)
      %val6_0 = pdl.result 0 of %op6  // txloss
      %val6_1 = pdl.result 1 of %op6  // txdelta
      pdl.replace %op5 with (%val6_1 : !pdl.value)  // tf.Mul
      pdl.erase %op4  // tf.Const
      pdl.erase %op3  // tf.PreventGradient
      pdl.replace %op2 with (%val6_0 : !pdl.value)  // tf.Mean
      pdl.erase %op1  // tf.Const
      pdl.erase %op0  // tf.SparseSoftmaxCrossEntropyWithLogits
    }
  }
}

// CHECK-LABEL: test.mlp_split
// CHECK: %[[FWD:.*]] = "kernel.FcFwd"(%arg0, %arg2) : (tensor<2x20xf32>, tensor<20x10xf32>) -> tensor<2x10xf32>
// CHECK: %[[SM:.*]]:2 = "kernel.SoftmaxCrossEntropy"(%[[FWD]], %arg1) : (tensor<2x10xf32>, tensor<2xi32>) -> (tensor<f32>, tensor<2x10xf32>)
// CHECK: %[[BWD:.*]] = "kernel.FcBwd"(%arg0, %[[SM]]#1, %arg2) : (tensor<2x20xf32>, tensor<2x10xf32>, tensor<20x10xf32>) -> tensor<20x10xf32>
// CHECK: return %[[SM:.*]]#0, %[[BWD]] : tensor<f32>, tensor<20x10xf32>
module @ir attributes { test.mlp_split } {
  func @main(%arg0: tensor<2x20xf32>, %arg1: tensor<2xi32>, %arg2: tensor<20x10xf32>) -> (tensor<f32>, tensor<20x10xf32>) {
    %0 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = "tf.Const"() {value = dense<1.000000e-01> : tensor<f32>} : () -> tensor<f32>
    %2 = "tf.Const"() {value = dense<5.000000e-01> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
    %3 = "tf.MatMul"(%arg0, %arg2) {transpose_a = false, transpose_b = false} : (tensor<2x20xf32>, tensor<20x10xf32>) -> tensor<2x10xf32>
    %loss, %backprop = "tf.SparseSoftmaxCrossEntropyWithLogits"(%3, %arg1) : (tensor<2x10xf32>, tensor<2xi32>) -> (tensor<2xf32>, tensor<2x10xf32>)
    %4 = "tf.Mean"(%loss, %0) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
    %5 = "tf.PreventGradient"(%backprop) : (tensor<2x10xf32>) -> tensor<2x10xf32>
    %6 = "tf.Mul"(%5, %2) : (tensor<2x10xf32>, tensor<2x1xf32>) -> tensor<2x10xf32>
    %7 = "tf.MatMul"(%arg0, %6) {transpose_a = true, transpose_b = false} : (tensor<2x20xf32>, tensor<2x10xf32>) -> tensor<20x10xf32>
    %8 = "tf.Mul"(%7, %1) : (tensor<20x10xf32>, tensor<f32>) -> tensor<20x10xf32>
    %9 = "tf.Sub"(%arg2, %8) : (tensor<20x10xf32>, tensor<20x10xf32>) -> tensor<20x10xf32>
    return %4, %9 : tensor<f32>, tensor<20x10xf32>
  }
}

