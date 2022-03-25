"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"()[^bb3] : () -> ()
  ^bb2:  // pred: ^bb0
    "cf.br"()[^bb3] : () -> ()
  ^bb3:  // 2 preds: ^bb1, ^bb2
    "func.return"() : () -> ()
  }) {function_type = (i1) -> (), sym_name = "func_condBranch"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    "cf.br"(%arg0)[^bb1] : (i32) -> ()
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = "arith.cmpi"(%0, %arg1) {predicate = 2 : i64} : (i32, i32) -> i1
    "cf.cond_br"(%1)[^bb2, ^bb3] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb2:  // pred: ^bb1
    %2 = "arith.constant"() {value = 1 : i32} : () -> i32
    %3 = "arith.addi"(%0, %2) : (i32, i32) -> i32
    "cf.br"(%3)[^bb1] : (i32) -> ()
  ^bb3:  // pred: ^bb1
    "func.return"() : () -> ()
  }) {function_type = (i32, i32) -> (), sym_name = "func_loop"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    "scf.for"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: index):
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index) -> (), sym_name = "nested_region"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    "scf.for"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: index):
      "scf.for"(%arg0, %arg1, %arg2) ({
      ^bb0(%arg4: index):
        "scf.for"(%arg0, %arg1, %arg2) ({
        ^bb0(%arg5: index):
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index) -> (), sym_name = "nested_region2"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32, %arg2: index, %arg3: index, %arg4: index):
    "cf.br"(%arg0)[^bb1] : (i32) -> ()
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = "arith.cmpi"(%0, %arg1) {predicate = 2 : i64} : (i32, i32) -> i1
    "cf.cond_br"(%1)[^bb2, ^bb3] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb2:  // pred: ^bb1
    %2 = "arith.constant"() {value = 1 : i32} : () -> i32
    %3 = "arith.addi"(%0, %2) : (i32, i32) -> i32
    "scf.for"(%arg2, %arg3, %arg4) ({
    ^bb0(%arg5: index):
      "scf.for"(%arg2, %arg3, %arg4) ({
      ^bb0(%arg6: index):
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "cf.br"(%3)[^bb1] : (i32) -> ()
  ^bb3:  // pred: ^bb1
    "func.return"() : () -> ()
  }) {function_type = (i32, i32, index, index, index) -> (), sym_name = "func_loop_nested_region"} : () -> ()
}) : () -> ()

// -----
