// RUN: mlir-opt --mlir-print-op-generic %s > %t.1 &&  MLIR %t.1 > %t.2 && mlir-opt --mlir-print-op-generic %t.2 > %t.3 && diff %t.1 %t.3
module {
   func @add() -> i32 {
      %z = constant 10 : i32
      br ^bb2
      ^bb2:
         return %z : i32
   }
}
