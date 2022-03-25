"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = true} : () -> i1
    %1 = "scf.if"(%0) ({
      %2 = "arith.constant"() {value = 1 : i32} : () -> i32
      "scf.yield"(%2) : (i32) -> ()
    }, {
      "scf.yield"(%arg0) : (i32) -> ()
    }) : (i1) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "simple"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    %0 = "scf.if"(%arg0) ({
      %1 = "arith.constant"() {value = 1 : i32} : () -> i32
      "scf.yield"(%1) : (i32) -> ()
    }, {
      %1 = "arith.constant"() {value = 1 : i32} : () -> i32
      "scf.yield"(%1) : (i32) -> ()
    }) : (i1) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i1) -> i32, sym_name = "simple_both_same"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i32):
    %0 = "scf.if"(%arg0) ({
      %1 = "arith.constant"() {value = 1 : i32} : () -> i32
      "scf.yield"(%1) : (i32) -> ()
    }, {
      "scf.yield"(%arg1) : (i32) -> ()
    }) : (i1) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i1, i32) -> i32, sym_name = "overdefined_unknown_condition"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    %0 = "scf.if"(%arg0) ({
      %1 = "arith.constant"() {value = 1 : i32} : () -> i32
      "scf.yield"(%1) : (i32) -> ()
    }, {
      %1 = "arith.constant"() {value = 2 : i32} : () -> i32
      "scf.yield"(%1) : (i32) -> ()
    }) : (i1) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i1) -> i32, sym_name = "overdefined_different_constants"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "scf.for"(%arg0, %arg1, %arg2, %0) ({
    ^bb0(%arg3: index, %arg4: i32):
      %2 = "arith.addi"(%arg4, %arg4) : (i32, i32) -> i32
      "scf.yield"(%2) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (index, index, index) -> i32, sym_name = "simple_loop"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "scf.for"(%arg0, %arg1, %arg2, %0) ({
    ^bb0(%arg3: index, %arg4: i32):
      %2 = "arith.addi"(%arg4, %arg4) : (i32, i32) -> i32
      "scf.yield"(%2) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (index, index, index) -> i32, sym_name = "loop_overdefined"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "scf.for"(%arg0, %arg1, %arg2, %0) ({
    ^bb0(%arg3: index, %arg4: i32):
      %2 = "arith.constant"() {value = 20 : i32} : () -> i32
      %3 = "arith.cmpi"(%arg4, %2) {predicate = 6 : i64} : (i32, i32) -> i1
      %4 = "scf.if"(%3) ({
        %5 = "arith.constant"() {value = 1 : i32} : () -> i32
        "scf.yield"(%5) : (i32) -> ()
      }, {
        %5 = "arith.addi"(%arg4, %0) : (i32, i32) -> i32
        "scf.yield"(%5) : (i32) -> ()
      }) : (i1) -> i32
      "scf.yield"(%4) : (i32) -> ()
    }) : (index, index, index, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (index, index, index) -> i32, sym_name = "loop_inner_control_flow"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 2 : i32} : () -> i32
    %1 = "scf.while"(%0) ({
    ^bb0(%arg1: i32):
      %2 = "arith.cmpi"(%arg1, %arg0) {predicate = 2 : i64} : (i32, i32) -> i1
      "scf.condition"(%2, %arg1) : (i1, i32) -> ()
    }, {
    ^bb0(%arg1: i32):
      "scf.yield"(%arg1) : (i32) -> ()
    }) : (i32) -> i32
    "func.return"() : () -> ()
  }) {function_type = (i32) -> (), sym_name = "loop_region_branch_terminator_op"} : () -> ()
}) : () -> ()

// -----
