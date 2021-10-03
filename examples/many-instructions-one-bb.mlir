// RUN: mlir-opt --mlir-print-op-generic %s> %t.1 &&  MLIR %t.1 > %t.2 && mlir-opt --mlir-print-op-generic %t.2 > %t.3 && diff %t.1 %t.3
module {
  func @add(%x: i32, %y: i32) -> i32 {
     %z = addi %x, %y : i32
     %c42 = constant 42 : i32
     %cond = cmpi "eq", %z, %c42 : i32
     %c10 = constant 10 : i32
     %c20 = constant 20 : i32
     return %z : i32
  }
}

