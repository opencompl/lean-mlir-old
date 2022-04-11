


"builtin.module"() ({
  "func.func"() ({
    %0 = "foo"() : () -> i32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.yield"() : () -> ()
    }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
    %1 = "arith.constant"() {value = 4 : index} : () -> index
    "affine.if"(%1) ({
      "affine.yield"() : () -> ()
    }, {
    }) {condition = #set} : (index) -> ()
    "foo.region"() ({
    ^bb0(%arg0: i32, %arg1: i32):
      %2 = "arith.addi"(%arg0, %arg0) : (i32, i32) -> i32
      "foo.yield"(%2) : (i32) -> ()
    }) : () -> ()
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "inline_notation"} : () -> ()
}) : () -> ()


