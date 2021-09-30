// module {
//    func @add() -> i32 {
//       %z = constant 10 : i32
//       br ^bb2
//       ^bb2:
//          return %z : i32
//    }
// }
// 

"builtin.module"() ( {
   // notice that this region has no entry BB.
  "builtin.func"() ( {
    // Notice that this region has no entry BB
    %0 = "std.constant"() {value = 10 : i32} : () -> i32
    "std.br"()[^bb1] : () -> ()
  ^bb1:  // pred: ^bb0
    "std.return"(%0) : (i32) -> ()
  }) {sym_name = "add", type = () -> i32} : () -> ()
}) : () -> ()


