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

// "module"() ( {
//   "module"() ( {
//     "pdl.pattern"() ( {
//       %0 = "pdl.type"() : () -> !pdl.type
//       %1 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
//       %2 = "pdl.operation"(%0) {attributeNames = [], name = "asm.zero", operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>} : (!pdl.type) -> !pdl.operation
//       %3 = "pdl.result"(%2) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
//       %4 = "pdl.operation"(%3, %1, %0) {attributeNames = [], name = "asm.add", operand_segment_sizes = dense<[2, 0, 1]> : vector<3xi32>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
//       "pdl.rewrite"(%4) ( {
//         "pdl.replace"(%4, %1) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : (!pdl.operation, !pdl.value) -> ()
//         "pdl.rewrite_end"() : () -> ()
//       }) : (!pdl.operation) -> ()
//     }) {benefit = 1 : i16} : () -> ()
//     "module_terminator"() : () -> ()
//   }) {sym_name = "patterns"} : () -> ()
//   "module"() ( {
//     "func"() ( {
//     ^bb0(%arg0: i32):  // no predecessors
//       %0 = "asm.zero"() : () -> i32
//       %1 = "asm.add"(%0, %arg0) : (i32, i32) -> i32
//       "std.return"(%1) : (i32) -> ()
//     }) {sym_name = "main", type = (i32) -> i32} : () -> ()
//     "module_terminator"() : () -> ()
//   }) {sym_name = "ir", test.mlp_split} : () -> ()
//   "module_terminator"() : () -> ()
// }) : () -> ()