// RUN: mlir-opt --mlir-print-op-generic %s> %t.1 &&  MLIR %t.1 > %t.2 && mlir-opt --mlir-print-op-generic %t.2 > %t.3 && diff %t.1 %t.3
// module {
//   func @add(%x: i32, %y: i32) -> i32 {
//      %z = addi %x, %y : i32
//      %c42 = constant 42 : i32
//      %cond = cmpi "eq", %z, %c42 : i32
//      %c10 = constant 10 : i32
//      %c20 = constant 20 : i32
//      return %z : i32
//   }
// }

"module"() ( {
  "func"() ( {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = "std.addi"(%arg0, %arg1) : (i32, i32) -> i32
    %c42_i32 = "std.constant"() {value = 42 : i32} : () -> i32
    %1 = "std.cmpi"(%0, %c42_i32) {predicate = 0 : i64} : (i32, i32) -> i1
    %c10_i32 = "std.constant"() {value = 10 : i32} : () -> i32
    %c20_i32 = "std.constant"() {value = 20 : i32} : () -> i32
    "std.return"(%0) : (i32) -> ()
  }) {sym_name = "add", type = (i32, i32) -> i32} : () -> ()
  "module_terminator"() : () -> ()
}) : () -> ()


