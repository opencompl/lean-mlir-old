// RUN: mlir-opt %s  -allow-unregistered-dialect -test-pdl-bytecode-pass -split-input-file | FileCheck %s

// -----

//===----------------------------------------------------------------------===//
// Asm add rewrite to write add (int 0) a -> a
//===----------------------------------------------------------------------===//

module @patterns {
  // fc_fwd
  pdl.pattern : benefit(1) {
    %ty = pdl.type 

    // right operand of add.
    %add_right_rand = pdl.operand : %ty
    // TODO: is pdl.operation  allowed to have empty arg list?
    // %zero_op = pdl.operation "asm.int" () {"value" = %int_val} -> (%c0_type : !pdl.type)
    %zero_op = pdl.operation "asm.zero" -> (%ty : !pdl.type)
    %zero_result = pdl.result 0 of %zero_op
    %add_op = pdl.operation "asm.add" (%zero_result, %add_right_rand : !pdl.value, !pdl.value) -> (%ty : !pdl.type)

    pdl.rewrite %add_op {
      // %op1 = pdl.operation "kernel.FcFwd" (%rxact, %weight : !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
      // vvv easy to make this crash! If we don't have the right number of results...
      /// vvv can our code prevent such errors?
      // %val1 = pdl.result 0 of %add_op
      pdl.replace %add_op with (%add_right_rand: !pdl.value)  
    }
  }
}

module @ir attributes { test.mlp_split } {
  func @main(%r: i32) -> (i32) {
    %c0 = "asm.zero"() : () -> i32
    %add = "asm.add"(%c0, %r) : (i32, i32) -> (i32)
    return %add : i32
  }
}

