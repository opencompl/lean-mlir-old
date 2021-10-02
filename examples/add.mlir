// RUN: mlir-opt --mlir-print-op-generic %s> %t.1 &&  MLIR %t.1 > %t.2 && mlir-opt --mlir-print-op-generic %t.2 > %t.3 && diff %t.1 %t.3
// module {
//   func @add(%x: i32, %y: i32) -> i32 {
//      %z = addi %x, %y : i32
//      return %z : i32
//   }
// }

"module"() ( {
  "func"() ( {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = "std.addi"(%arg0, %arg1) : (i32, i32) -> i32
    "std.return"(%0) : (i32) -> ()
  }) {sym_name = "add", type = (i32, i32) -> i32} : () -> ()
  "module_terminator"() : () -> ()
}) : () -> ()
