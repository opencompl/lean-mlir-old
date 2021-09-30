// module {
//   func @add(%x: i32, %y: i32) -> i32 {
//      %z = addi %x, %y : i32
//      return %z : i32
//   }
// }

"builtin.module"() ( {
  "builtin.func"() ( {
  ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
    %0 = "std.addi"(%arg0, %arg1) : (i32, i32) -> i32
    "std.return"(%0) : (i32) -> ()
  }) {sym_name = "add", type = (i32, i32) -> i32} : () -> ()
}) : () -> ()

