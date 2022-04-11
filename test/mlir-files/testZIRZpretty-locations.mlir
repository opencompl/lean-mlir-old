


"builtin.module"() ({
  "func.func"() ({
    %0 = "foo"() : () -> i32
    %1 = "arith.constant"() {value = 4 : index} : () -> index
    %2 = "arith.constant"() {value = 4 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.yield"() : () -> ()
    }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
    "affine.if"(%1) ({
      "affine.yield"() : () -> ()
    }, {
    }) {condition = #set} : (index) -> ()
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "inline_notation"} : () -> ()
}) : () -> ()


