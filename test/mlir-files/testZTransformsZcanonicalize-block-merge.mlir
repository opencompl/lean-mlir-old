"builtin.module"() ({
  "func.func"() ({
    "foo.cond_br"()[^bb1, ^bb2] : () -> ()
  ^bb1:  // pred: ^bb0
    "func.return"() : () -> ()
  ^bb2:  // pred: ^bb0
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "return_blocks"} : () -> ()
  "func.func"() ({
    "foo.cond_br"()[^bb1, ^bb2] : () -> ()
  ^bb1(%0: i32):  // pred: ^bb0
    "func.return"(%0) : (i32) -> ()
  ^bb2(%1: i32):  // pred: ^bb0
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "matching_arguments"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    "foo.cond_br"()[^bb1, ^bb2] : () -> ()
  ^bb1:  // pred: ^bb0
    "func.return"(%arg0) : (i32) -> ()
  ^bb2:  // pred: ^bb0
    "func.return"(%arg1) : (i32) -> ()
  }) {function_type = (i32, i32) -> i32, sym_name = "mismatch_unknown_terminator"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i32, %arg2: i32):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "func.return"(%arg1) : (i32) -> ()
  ^bb2:  // pred: ^bb0
    "func.return"(%arg2) : (i32) -> ()
  }) {function_type = (i1, i32, i32) -> i32, sym_name = "mismatch_operands"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i32, %arg2: i32):
    "cf.cond_br"(%arg0, %arg2, %arg1)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, i32, i32) -> ()
  ^bb1(%0: i32):  // pred: ^bb0
    "func.return"(%arg1, %0) : (i32, i32) -> ()
  ^bb2(%1: i32):  // pred: ^bb0
    "func.return"(%arg2, %1) : (i32, i32) -> ()
  }) {function_type = (i1, i32, i32) -> (i32, i32), sym_name = "mismatch_operands_matching_arguments"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i32, %arg2: i32):
    "cf.cond_br"(%arg0, %arg2, %arg1)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, i32, i32) -> ()
  ^bb1(%0: i32):  // pred: ^bb0
    "func.return"(%arg1, %0) : (i32, i32) -> ()
  ^bb2(%1: i32):  // pred: ^bb0
    "func.return"(%1, %arg2) : (i32, i32) -> ()
  }) {function_type = (i1, i32, i32) -> (i32, i32), sym_name = "mismatch_argument_uses"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i32, %arg2: i16):
    "cf.cond_br"(%arg0, %arg1, %arg2)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, i32, i16) -> ()
  ^bb1(%0: i32):  // pred: ^bb0
    "foo.return"(%0) : (i32) -> ()
  ^bb2(%1: i16):  // pred: ^bb0
    "foo.return"(%1) : (i16) -> ()
  }) {function_type = (i1, i32, i16) -> (), sym_name = "mismatch_argument_types"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i32):
    "cf.cond_br"(%arg0, %arg1)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3 × i32>} : (i1, i32) -> ()
  ^bb1(%0: i32):  // pred: ^bb0
    "foo.return"(%0) : (i32) -> ()
  ^bb2:  // pred: ^bb0
    "foo.return"() : () -> ()
  }) {function_type = (i1, i32) -> (), sym_name = "mismatch_argument_count"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "foo.return"() : () -> ()
  ^bb2:  // pred: ^bb0
    "func.return"() : () -> ()
  }) {function_type = (i1) -> (), sym_name = "mismatch_operations"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "foo.op"() : () -> ()
    "func.return"() : () -> ()
  ^bb2:  // pred: ^bb0
    "func.return"() : () -> ()
  }) {function_type = (i1) -> (), sym_name = "mismatch_operation_count"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "scf.if"(%arg0) ({
      "foo.op"() : () -> ()
      "scf.yield"() : () -> ()
    }, {
    }) : (i1) -> ()
    "func.return"() : () -> ()
  ^bb2:  // pred: ^bb0
    "scf.if"(%arg0) ({
      "foo.op"() : () -> ()
      "scf.yield"() : () -> ()
    }, {
    }) : (i1) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1) -> (), sym_name = "contains_regions"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i1):
    %0 = "foo.op"() : () -> i1
    "cf.cond_br"(%arg0)[^bb2, ^bb3] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // 2 preds: ^bb1, ^bb2
    "cf.cond_br"(%0)[^bb1, ^bb3] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb2:  // pred: ^bb0
    "cf.cond_br"(%arg1)[^bb1, ^bb3] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
    "func.return"() : () -> ()
  }) {function_type = (i1, i1) -> (), sym_name = "mismatch_loop"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<i32>, %arg2: memref<i1>):
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.constant"() {value = true} : () -> i1
    "cf.br"()[^bb1] : () -> ()
  ^bb1:  // 3 preds: ^bb0, ^bb2, ^bb3
    "cf.cond_br"(%arg0)[^bb2, ^bb3] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb2:  // pred: ^bb1
    "memref.store"(%0, %arg1) : (i32, memref<i32>) -> ()
    "cf.br"()[^bb1] : () -> ()
  ^bb3:  // pred: ^bb1
    "memref.store"(%1, %arg2) : (i1, memref<i1>) -> ()
    "cf.br"()[^bb1] : () -> ()
  }) {function_type = (i1, memref<i32>, memref<i1>) -> (), sym_name = "mismatch_operand_types"} : () -> ()
  "func.func"() ({
  }) {function_type = (i32, i32) -> (), sym_name = "print", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.cmpi"(%arg1, %arg0) {predicate = 2 : i64} : (i32, i32) -> i1
    "cf.cond_br"(%1)[^bb1, ^bb4] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    %2 = "arith.addi"(%arg1, %0) : (i32, i32) -> i32
    "cf.br"(%2)[^bb3] : (i32) -> ()
  ^bb2:  // pred: ^bb3
    %3 = "arith.addi"(%4, %0) : (i32, i32) -> i32
    "cf.br"(%3)[^bb3] : (i32) -> ()
  ^bb3(%4: i32):  // 2 preds: ^bb1, ^bb2
    %5 = "arith.cmpi"(%4, %arg0) {predicate = 2 : i64} : (i32, i32) -> i1
    "func.call"(%4, %2) {callee = @print} : (i32, i32) -> ()
    "cf.cond_br"(%5)[^bb2, ^bb4] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb4:  // 2 preds: ^bb0, ^bb3
    "func.return"() : () -> ()
  }) {function_type = (i32, i32) -> (), sym_name = "nomerge"} : () -> ()
  "func.func"() ({
    %0 = "test.producing_br"()[^bb1, ^bb2] {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> i32
  ^bb1:  // pred: ^bb0
    "test.br"(%0)[^bb4] : (i32) -> ()
  ^bb2:  // pred: ^bb0
    %1 = "foo.def"() : () -> i32
    "test.br"()[^bb3] : () -> ()
  ^bb3:  // pred: ^bb2
    "test.br"(%1)[^bb4] : (i32) -> ()
  ^bb4(%2: i32):  // 2 preds: ^bb1, ^bb3
    "func.return"(%2) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "mismatch_dominance"} : () -> ()
}) : () -> ()

// -----
