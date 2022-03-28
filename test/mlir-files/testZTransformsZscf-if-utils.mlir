"builtin.module"() ({
^bb0:
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: index, %arg2: memref<? × f32>, %arg3: i8):
    %0 = "scf.if"(%arg0) ({
      %1 = "some_op"(%arg0, %arg2) : (i1, memref<? × f32>) -> i8
      "scf.yield"(%1) : (i8) -> ()
    }, {
      "scf.yield"(%arg3) : (i8) -> ()
    }) : (i1) -> i8
    "func.return"() : () -> ()
  }) {function_type = (i1, index, memref<? × f32>, i8) -> (), sym_name = "outline_if_else"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: index, %arg2: memref<? × f32>, %arg3: i8):
    "scf.if"(%arg0) ({
      "some_op"(%arg0, %arg2) : (i1, memref<? × f32>) -> ()
      "scf.yield"() : () -> ()
    }, {
    }) : (i1) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, index, memref<? × f32>, i8) -> (), sym_name = "outline_if"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: index, %arg2: memref<? × f32>, %arg3: i8):
    "scf.if"(%arg0) ({
      "scf.yield"() : () -> ()
    }, {
      "some_op"(%arg0, %arg2) : (i1, memref<? × f32>) -> ()
      "scf.yield"() : () -> ()
    }) : (i1) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, index, memref<? × f32>, i8) -> (), sym_name = "outline_empty_if_else"} : () -> ()
}) : () -> ()


