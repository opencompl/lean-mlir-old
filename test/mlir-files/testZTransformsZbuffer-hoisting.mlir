"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%arg1)[^bb3] : (memref<2 × f32>) -> ()
  ^bb2:  // pred: ^bb0
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.br"(%0)[^bb3] : (memref<2 × f32>) -> ()
  ^bb3(%1: memref<2 × f32>):  // 2 preds: ^bb1, ^bb2
    "test.copy"(%1, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "condBranch"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<? × f32>, %arg2: memref<? × f32>, %arg3: index):
    "cf.cond_br"(%arg0, %arg3)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3 × i32>} : (i1, index) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%arg1)[^bb3] : (memref<? × f32>) -> ()
  ^bb2(%0: index):  // pred: ^bb0
    %1 = "memref.alloc"(%0) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
    "test.buffer_based"(%arg1, %1) : (memref<? × f32>, memref<? × f32>) -> ()
    "cf.br"(%1)[^bb3] : (memref<? × f32>) -> ()
  ^bb3(%2: memref<? × f32>):  // 2 preds: ^bb1, ^bb2
    "test.copy"(%2, %arg2) : (memref<? × f32>, memref<? × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<? × f32>, memref<? × f32>, index) -> (), sym_name = "condBranchDynamicType"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<? × f32>, %arg2: memref<? × f32>, %arg3: index):
    "cf.cond_br"(%arg0, %arg3)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3 × i32>} : (i1, index) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%arg1)[^bb6] : (memref<? × f32>) -> ()
  ^bb2(%0: index):  // pred: ^bb0
    %1 = "memref.alloc"(%0) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
    "test.buffer_based"(%arg1, %1) : (memref<? × f32>, memref<? × f32>) -> ()
    "cf.cond_br"(%arg0)[^bb3, ^bb4] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb3:  // pred: ^bb2
    "cf.br"(%1)[^bb5] : (memref<? × f32>) -> ()
  ^bb4:  // pred: ^bb2
    "cf.br"(%1)[^bb5] : (memref<? × f32>) -> ()
  ^bb5(%2: memref<? × f32>):  // 2 preds: ^bb3, ^bb4
    "cf.br"(%2)[^bb6] : (memref<? × f32>) -> ()
  ^bb6(%3: memref<? × f32>):  // 2 preds: ^bb1, ^bb5
    "cf.br"(%3)[^bb7] : (memref<? × f32>) -> ()
  ^bb7(%4: memref<? × f32>):  // pred: ^bb6
    "test.copy"(%4, %arg2) : (memref<? × f32>, memref<? × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<? × f32>, memref<? × f32>, index) -> (), sym_name = "condBranchDynamicTypeNested"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "cf.cond_br"(%arg0, %arg1)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3 × i32>} : (i1, memref<2 × f32>) -> ()
  ^bb1:  // pred: ^bb0
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.br"(%0)[^bb2] : (memref<2 × f32>) -> ()
  ^bb2(%1: memref<2 × f32>):  // 2 preds: ^bb0, ^bb1
    "test.copy"(%1, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "criticalEdge"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.cond_br"(%arg0, %arg1, %0, %0, %arg1)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 2, 2]> : vector<3 × i32>} : (i1, memref<2 × f32>, memref<2 × f32>, memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb1(%1: memref<2 × f32>, %2: memref<2 × f32>):  // pred: ^bb0
    "cf.br"(%1, %2)[^bb3] : (memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb2(%3: memref<2 × f32>, %4: memref<2 × f32>):  // pred: ^bb0
    "cf.br"(%3, %4)[^bb3] : (memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb3(%5: memref<2 × f32>, %6: memref<2 × f32>):  // 2 preds: ^bb1, ^bb2
    %7 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%7, %7) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "test.copy"(%7, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "ifElse"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.cond_br"(%arg0, %arg1, %0, %0, %arg1)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 2, 2]> : vector<3 × i32>} : (i1, memref<2 × f32>, memref<2 × f32>, memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb1(%1: memref<2 × f32>, %2: memref<2 × f32>):  // pred: ^bb0
    "cf.br"(%1, %2)[^bb3] : (memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb2(%3: memref<2 × f32>, %4: memref<2 × f32>):  // pred: ^bb0
    "cf.br"(%3, %4)[^bb3] : (memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb3(%5: memref<2 × f32>, %6: memref<2 × f32>):  // 2 preds: ^bb1, ^bb2
    "test.copy"(%arg1, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "ifElseNoUsers"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.cond_br"(%arg0, %arg1, %0, %0, %arg1)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 2, 2]> : vector<3 × i32>} : (i1, memref<2 × f32>, memref<2 × f32>, memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb1(%1: memref<2 × f32>, %2: memref<2 × f32>):  // pred: ^bb0
    "cf.br"(%1, %2)[^bb5] : (memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb2(%3: memref<2 × f32>, %4: memref<2 × f32>):  // pred: ^bb0
    "cf.cond_br"(%arg0, %3, %4)[^bb3, ^bb4] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb3(%5: memref<2 × f32>):  // pred: ^bb2
    "cf.br"(%5, %3)[^bb5] : (memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb4(%6: memref<2 × f32>):  // pred: ^bb2
    "cf.br"(%3, %6)[^bb5] : (memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb5(%7: memref<2 × f32>, %8: memref<2 × f32>):  // 3 preds: ^bb1, ^bb3, ^bb4
    %9 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%7, %9) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "test.copy"(%9, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "ifElseNested"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg0, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%0, %1) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × f32>) -> (), sym_name = "redundantOperations"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.br"(%0)[^bb3] : (memref<2 × f32>) -> ()
  ^bb2:  // pred: ^bb0
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %1) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.br"(%1)[^bb3] : (memref<2 × f32>) -> ()
  ^bb3(%2: memref<2 × f32>):  // 2 preds: ^bb1, ^bb2
    "test.copy"(%2, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "moving_alloc_and_inserting_missing_dealloc"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%arg1)[^bb3] : (memref<2 × f32>) -> ()
  ^bb2:  // pred: ^bb0
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "memref.dealloc"(%0) : (memref<2 × f32>) -> ()
    "cf.br"(%0)[^bb3] : (memref<2 × f32>) -> ()
  ^bb3(%1: memref<2 × f32>):  // 2 preds: ^bb1, ^bb2
    "test.copy"(%1, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "moving_invalid_dealloc_op_complex"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%arg1)[^bb3] : (memref<2 × f32>) -> ()
  ^bb2:  // pred: ^bb0
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.region_buffer_based"(%arg1, %0) ({
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
      "test.buffer_based"(%arg1, %2) : (memref<2 × f32>, memref<2 × f32>) -> ()
      %3 = "math.exp"(%arg3) : (f32) -> f32
      "test.region_yield"(%3) : (f32) -> ()
    }) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.br"(%0)[^bb3] : (memref<2 × f32>) -> ()
  ^bb3(%1: memref<2 × f32>):  // 2 preds: ^bb1, ^bb2
    "test.copy"(%1, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "nested_regions_and_cond_branch"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "arith.cmpi"(%arg0, %arg1) {predicate = 0 : i64} : (index, index) -> i1
    %1 = "memref.alloc"(%arg0, %arg0) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
    %2 = "scf.if"(%0) ({
      "scf.yield"(%1) : (memref<?x?xf32>) -> ()
    }, {
      %3 = "memref.alloc"(%arg0, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
      "scf.yield"(%1) : (memref<?x?xf32>) -> ()
    }) : (i1) -> memref<?x?xf32>
    "func.return"(%2) : (memref<?x?xf32>) -> ()
  }) {function_type = (index, index) -> memref<?x?xf32>, sym_name = "nested_region_control_flow"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "arith.cmpi"(%arg0, %arg1) {predicate = 0 : i64} : (index, index) -> i1
    %1 = "memref.alloc"(%arg0, %arg0) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
    %2 = "scf.if"(%0) ({
      "scf.yield"(%1) : (memref<?x?xf32>) -> ()
    }, {
      %3 = "memref.alloc"(%arg0, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
      "scf.yield"(%3) : (memref<?x?xf32>) -> ()
    }) : (i1) -> memref<?x?xf32>
    "func.return"(%2) : (memref<?x?xf32>) -> ()
  }) {function_type = (index, index) -> memref<?x?xf32>, sym_name = "nested_region_control_flow_div"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "arith.cmpi"(%arg0, %arg1) {predicate = 0 : i64} : (index, index) -> i1
    %1 = "memref.alloc"(%arg0, %arg0) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
    %2 = "scf.if"(%0) ({
      %3 = "scf.if"(%0) ({
        "scf.yield"(%1) : (memref<?x?xf32>) -> ()
      }, {
        %4 = "memref.alloc"(%arg0, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
        "scf.yield"(%4) : (memref<?x?xf32>) -> ()
      }) : (i1) -> memref<?x?xf32>
      "scf.yield"(%3) : (memref<?x?xf32>) -> ()
    }, {
      %3 = "memref.alloc"(%arg1, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
      "scf.yield"(%3) : (memref<?x?xf32>) -> ()
    }) : (i1) -> memref<?x?xf32>
    "func.return"(%2) : (memref<?x?xf32>) -> ()
  }) {function_type = (index, index) -> memref<?x?xf32>, sym_name = "nested_region_control_flow_div_nested"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i1, %arg2: index):
    %0 = "scf.if"(%arg1) ({
      %1 = "arith.constant"() {value = 1 : i32} : () -> i32
      %2 = "arith.addi"(%arg0, %1) : (i32, i32) -> i32
      %3 = "arith.index_cast"(%2) : (i32) -> index
      %4 = "memref.alloc"(%arg2, %3) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
      "scf.yield"(%4) : (memref<?x?xf32>) -> ()
    }, {
      %1 = "arith.constant"() {value = 2 : i32} : () -> i32
      %2 = "arith.addi"(%arg0, %1) : (i32, i32) -> i32
      %3 = "arith.index_cast"(%2) : (i32) -> index
      %4 = "memref.alloc"(%arg2, %3) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
      "scf.yield"(%4) : (memref<?x?xf32>) -> ()
    }) : (i1) -> memref<?x?xf32>
    "func.return"(%0) : (memref<?x?xf32>) -> ()
  }) {function_type = (i32, i1, index) -> memref<?x?xf32>, sym_name = "nested_region_control_flow_div_nested_dependencies"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "memref.alloc"(%arg0, %arg0) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
    %1 = "test.region_if"(%0) ({
    ^bb0(%arg1: memref<?x?xf32>):
      "test.region_if_yield"(%arg1) : (memref<?x?xf32>) -> ()
    }, {
    ^bb0(%arg1: memref<?x?xf32>):
      "test.region_if_yield"(%arg1) : (memref<?x?xf32>) -> ()
    }, {
    ^bb0(%arg1: memref<?x?xf32>):
      "test.region_if_yield"(%arg1) : (memref<?x?xf32>) -> ()
    }) : (memref<?x?xf32>) -> memref<?x?xf32>
    "func.return"(%1) : (memref<?x?xf32>) -> ()
  }) {function_type = (index) -> memref<?x?xf32>, sym_name = "inner_region_control_flow"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "memref.alloc"(%arg0, %arg0) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
    %1 = "test.region_if"(%0) ({
    ^bb0(%arg2: memref<?x?xf32>):
      "test.region_if_yield"(%arg2) : (memref<?x?xf32>) -> ()
    }, {
    ^bb0(%arg2: memref<?x?xf32>):
      %2 = "memref.alloc"(%arg0, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<?x?xf32>
      "test.region_if_yield"(%2) : (memref<?x?xf32>) -> ()
    }, {
    ^bb0(%arg2: memref<?x?xf32>):
      "test.region_if_yield"(%arg2) : (memref<?x?xf32>) -> ()
    }) : (memref<?x?xf32>) -> memref<?x?xf32>
    "func.return"(%1) : (memref<?x?xf32>) -> ()
  }) {function_type = (index, index) -> memref<?x?xf32>, sym_name = "inner_region_control_flow_div"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%arg1)[^bb3] : (memref<2 × f32>) -> ()
  ^bb2:  // pred: ^bb0
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.br"(%0)[^bb3] : (memref<2 × f32>) -> ()
  ^bb3(%1: memref<2 × f32>):  // 2 preds: ^bb1, ^bb2
    "test.copy"(%1, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "condBranchAlloca"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.cond_br"(%arg0, %arg1, %0, %0, %arg1)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 2, 2]> : vector<3 × i32>} : (i1, memref<2 × f32>, memref<2 × f32>, memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb1(%1: memref<2 × f32>, %2: memref<2 × f32>):  // pred: ^bb0
    "cf.br"(%1, %2)[^bb5] : (memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb2(%3: memref<2 × f32>, %4: memref<2 × f32>):  // pred: ^bb0
    "cf.cond_br"(%arg0, %3, %4)[^bb3, ^bb4] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb3(%5: memref<2 × f32>):  // pred: ^bb2
    "cf.br"(%5, %3)[^bb5] : (memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb4(%6: memref<2 × f32>):  // pred: ^bb2
    "cf.br"(%3, %6)[^bb5] : (memref<2 × f32>, memref<2 × f32>) -> ()
  ^bb5(%7: memref<2 × f32>, %8: memref<2 × f32>):  // 3 preds: ^bb1, ^bb3, ^bb4
    %9 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%7, %9) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "test.copy"(%9, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "ifElseNestedAlloca"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  // pred: ^bb0
    "cf.br"(%arg1)[^bb3] : (memref<2 × f32>) -> ()
  ^bb2:  // pred: ^bb0
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.region_buffer_based"(%arg1, %0) ({
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
      "test.buffer_based"(%arg1, %2) : (memref<2 × f32>, memref<2 × f32>) -> ()
      %3 = "math.exp"(%arg3) : (f32) -> f32
      "test.region_yield"(%3) : (f32) -> ()
    }) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.br"(%0)[^bb3] : (memref<2 × f32>) -> ()
  ^bb3(%1: memref<2 × f32>):  // 2 preds: ^bb1, ^bb2
    "test.copy"(%1, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "nestedRegionsAndCondBranchAlloca"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    %1 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg5: index, %arg6: memref<2 × f32>):
      %2 = "arith.cmpi"(%arg5, %arg1) {predicate = 0 : i64} : (index, index) -> i1
      %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
      "scf.yield"(%3) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "test.copy"(%1, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "loop_alloc"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    %1 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg4: index, %arg5: memref<2 × f32>):
      %2 = "arith.cmpi"(%arg4, %arg1) {predicate = 0 : i64} : (index, index) -> i1
      %3 = "scf.if"(%2) ({
        %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
        "scf.yield"(%4) : (memref<2 × f32>) -> ()
      }, {
        "scf.yield"(%0) : (memref<2 × f32>) -> ()
      }) : (i1) -> memref<2 × f32>
      "scf.yield"(%3) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "func.return"(%1) : (memref<2 × f32>) -> ()
  }) {function_type = (index, index, index, memref<2 × f32>) -> memref<2 × f32>, sym_name = "loop_nested_if_alloc"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    %1 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg5: index, %arg6: memref<2 × f32>):
      %2 = "scf.for"(%arg0, %arg1, %arg2, %arg6) ({
      ^bb0(%arg7: index, %arg8: memref<2 × f32>):
        %3 = "scf.for"(%arg0, %arg1, %arg2, %arg8) ({
        ^bb0(%arg9: index, %arg10: memref<2 × f32>):
          %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
          %5 = "arith.cmpi"(%arg5, %arg1) {predicate = 0 : i64} : (index, index) -> i1
          %6 = "scf.if"(%5) ({
            %7 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
            "scf.yield"(%7) : (memref<2 × f32>) -> ()
          }, {
            "scf.yield"(%arg10) : (memref<2 × f32>) -> ()
          }) : (i1) -> memref<2 × f32>
          "scf.yield"(%6) : (memref<2 × f32>) -> ()
        }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
        "scf.yield"(%3) : (memref<2 × f32>) -> ()
      }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
      "scf.yield"(%2) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "test.copy"(%1, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "loop_nested_alloc"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<? × f32>, %arg5: memref<? × f32>):
    %0 = "memref.alloc"(%arg3) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
    %1 = "scf.for"(%arg0, %arg1, %arg2, %arg4) ({
    ^bb0(%arg6: index, %arg7: memref<? × f32>):
      %2 = "scf.for"(%arg0, %arg1, %arg2, %arg7) ({
      ^bb0(%arg8: index, %arg9: memref<? × f32>):
        %3 = "scf.for"(%arg0, %arg1, %arg2, %arg9) ({
        ^bb0(%arg10: index, %arg11: memref<? × f32>):
          %4 = "arith.cmpi"(%arg6, %arg1) {predicate = 0 : i64} : (index, index) -> i1
          %5 = "scf.if"(%4) ({
            %6 = "memref.alloc"(%arg10) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
            "scf.yield"(%6) : (memref<? × f32>) -> ()
          }, {
            "scf.yield"(%arg11) : (memref<? × f32>) -> ()
          }) : (i1) -> memref<? × f32>
          "scf.yield"(%5) : (memref<? × f32>) -> ()
        }) : (index, index, index, memref<? × f32>) -> memref<? × f32>
        "scf.yield"(%3) : (memref<? × f32>) -> ()
      }) : (index, index, index, memref<? × f32>) -> memref<? × f32>
      "scf.yield"(%0) : (memref<? × f32>) -> ()
    }) : (index, index, index, memref<? × f32>) -> memref<? × f32>
    "test.copy"(%1, %arg5) : (memref<? × f32>, memref<? × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, index, memref<? × f32>, memref<? × f32>) -> (), sym_name = "loop_nested_alloc_dyn_dependency"} : () -> ()
}) : () -> ()

// -----
