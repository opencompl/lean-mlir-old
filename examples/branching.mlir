// module {
//   func @add(%x: i32, %y: i32) -> i32 {
//      %z = addi %x, %y : i32
//      %c42 = constant 42 : i32
//      %cond = cmpi "eq", %z, %c42 : i32
//      %c10 = constant 10 : i32
//      %c20 = constant 20 : i32
//      cond_br %cond, ^truebb(%c10 : i32), ^falsebb(%c20 : i32)
// 
//     ^truebb(%ptrue: i32):
//       %outt = addi %ptrue, %x : i32
//       return %outt : i32
// 
//     ^falsebb(%pfalse: i32):
//       %outf = addi %pfalse, %y : i32
//       return %outf : i32
//   }
// }

"func"() ( {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "std.addi"(%arg0, %arg1) : (i32, i32) -> i32
    %c42_i32 = "std.constant"() {value = 42 : i32} : () -> i32
     %1 = "std.cmpi"(%0, %c42_i32) {predicate = 0 : i64} : (i32, i32) -> i1
     %c10_i32 = "std.constant"() {value = 10 : i32} : () -> i32
     %c20_i32 = "std.constant"() {value = 20 : i32} : () -> i32
     "std.cond_br"(%1, %c10_i32, %c20_i32)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3xi32>} : (i1, i32, i32) -> ()
   ^bb1(%2: i32):
     %3 = "std.addi"(%2, %arg0) : (i32, i32) -> i32
     "std.return"(%3) : (i32) -> ()
   ^bb2(%4: i32):
     %5 = "std.addi"(%4, %arg1) : (i32, i32) -> i32
     "std.return"(%5) : (i32) -> ()
  }) {sym_name = "add", type = (i32, i32) -> i32} : () -> ()
  "module_terminator"() : () -> ()
}) : () -

