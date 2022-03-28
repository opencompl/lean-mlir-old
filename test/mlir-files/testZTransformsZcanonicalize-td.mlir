"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1):
    "scf.if"(%arg0) ({
      %0 = "arith.constant"() {value = 1 : i32} : () -> i32
      %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
      "foo.yield"(%1) : (i32) -> ()
      "scf.yield"() : () -> ()
    }, {
    }) : (i1) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1) -> (), sym_name = "default_insertion_position"} : () -> ()
  "func.func"() ({
    "test.one_region_op"() ({
      %0 = "arith.constant"() {value = 1 : i32} : () -> i32
      %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
      "foo.yield"(%1) : (i32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "custom_insertion_position"} : () -> ()
}) : () -> ()


