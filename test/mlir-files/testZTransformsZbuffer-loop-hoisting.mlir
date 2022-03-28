"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  
    "cf.br"(%arg1)[^bb3] : (memref<2 × f32>) -> ()
  ^bb2:  
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.buffer_based"(%arg1, %0) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.br"(%0)[^bb3] : (memref<2 × f32>) -> ()
  ^bb3(%1: memref<2 × f32>):  
    "test.copy"(%1, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "condBranch"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<? × f32>, %arg2: memref<? × f32>, %arg3: index):
    "cf.cond_br"(%arg0, %arg3)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3 × i32>} : (i1, index) -> ()
  ^bb1:  
    "cf.br"(%arg1)[^bb3] : (memref<? × f32>) -> ()
  ^bb2(%0: index):  
    %1 = "memref.alloc"(%0) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
    "test.buffer_based"(%arg1, %1) : (memref<? × f32>, memref<? × f32>) -> ()
    "cf.br"(%1)[^bb3] : (memref<? × f32>) -> ()
  ^bb3(%2: memref<? × f32>):  
    "test.copy"(%2, %arg2) : (memref<? × f32>, memref<? × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<? × f32>, memref<? × f32>, index) -> (), sym_name = "condBranchDynamicType"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  
    "cf.br"(%arg1)[^bb3] : (memref<2 × f32>) -> ()
  ^bb2:  
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "test.region_buffer_based"(%arg1, %0) ({
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
      "test.buffer_based"(%arg1, %2) : (memref<2 × f32>, memref<2 × f32>) -> ()
      %3 = "math.exp"(%arg3) : (f32) -> f32
      "test.region_yield"(%3) : (f32) -> ()
    }) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "cf.br"(%0)[^bb3] : (memref<2 × f32>) -> ()
  ^bb3(%1: memref<2 × f32>):  
    "test.copy"(%1, %arg2) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "nested_regions_and_cond_branch"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "arith.cmpi"(%arg0, %arg1) {predicate = 0 : i64} : (index, index) -> i1
    %1 = "memref.alloc"(%arg0, %arg0) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    %2 = "scf.if"(%0) ({
      "scf.yield"(%1) : (memref<? × ? × f32>) -> ()
    }, {
      %3 = "memref.alloc"(%arg0, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
      "scf.yield"(%1) : (memref<? × ? × f32>) -> ()
    }) : (i1) -> memref<? × ? × f32>
    "func.return"(%2) : (memref<? × ? × f32>) -> ()
  }) {function_type = (index, index) -> memref<? × ? × f32>, sym_name = "nested_region_control_flow"} : () -> ()
}) : () -> ()


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
            %8 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
            %9 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
            "scf.yield"(%9) : (memref<2 × f32>) -> ()
          }, {
            "scf.yield"(%arg10) : (memref<2 × f32>) -> ()
          }) : (i1) -> memref<2 × f32>
          %7 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
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
          %4 = "memref.alloc"(%arg10) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
          %5 = "arith.cmpi"(%arg6, %arg1) {predicate = 0 : i64} : (index, index) -> i1
          %6 = "scf.if"(%5) ({
            %8 = "memref.alloc"(%arg10) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
            "scf.yield"(%8) : (memref<? × f32>) -> ()
          }, {
            "scf.yield"(%arg11) : (memref<? × f32>) -> ()
          }) : (i1) -> memref<? × f32>
          %7 = "memref.alloc"(%arg10) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
          "scf.yield"(%6) : (memref<? × f32>) -> ()
        }) : (index, index, index, memref<? × f32>) -> memref<? × f32>
        "scf.yield"(%3) : (memref<? × f32>) -> ()
      }) : (index, index, index, memref<? × f32>) -> memref<? × f32>
      "scf.yield"(%0) : (memref<? × f32>) -> ()
    }) : (index, index, index, memref<? × f32>) -> memref<? × f32>
    "test.copy"(%1, %arg5) : (memref<? × f32>, memref<? × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, index, memref<? × f32>, memref<? × f32>) -> (), sym_name = "loop_nested_alloc_dyn_dependency"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    %1 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg5: index, %arg6: memref<2 × f32>):
      %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
      "scf.yield"(%0) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "test.copy"(%1, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "hoist_one_loop"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg5: index, %arg6: memref<2 × f32>):
      %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
      "scf.yield"(%1) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "test.copy"(%0, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "no_hoist_one_loop"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    %1 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg5: index, %arg6: memref<2 × f32>):
      %2 = "scf.for"(%arg0, %arg1, %arg2, %arg6) ({
      ^bb0(%arg7: index, %arg8: memref<2 × f32>):
        %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
        "scf.yield"(%0) : (memref<2 × f32>) -> ()
      }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
      "scf.yield"(%0) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "test.copy"(%1, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "hoist_multiple_loop"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg5: index, %arg6: memref<2 × f32>):
      %1 = "arith.cmpi"(%arg5, %arg1) {predicate = 0 : i64} : (index, index) -> i1
      %2 = "scf.if"(%1) ({
        %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
        "scf.yield"(%3) : (memref<2 × f32>) -> ()
      }, {
        "scf.yield"(%arg6) : (memref<2 × f32>) -> ()
      }) : (i1) -> memref<2 × f32>
      "scf.yield"(%2) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "test.copy"(%0, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "no_hoist_one_loop_conditional"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    %1 = "arith.cmpi"(%arg0, %arg1) {predicate = 0 : i64} : (index, index) -> i1
    %2 = "scf.if"(%1) ({
      %3 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
      ^bb0(%arg5: index, %arg6: memref<2 × f32>):
        %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
        "scf.yield"(%0) : (memref<2 × f32>) -> ()
      }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
      "scf.yield"(%0) : (memref<2 × f32>) -> ()
    }, {
      "scf.yield"(%0) : (memref<2 × f32>) -> ()
    }) : (i1) -> memref<2 × f32>
    "test.copy"(%2, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "hoist_one_loop_conditional"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    %1 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg5: index, %arg6: memref<2 × f32>):
      %2 = "memref.alloc"(%arg5) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
      "scf.yield"(%0) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "test.copy"(%1, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "no_hoist_one_loop_dependency"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    %1 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg5: index, %arg6: memref<2 × f32>):
      %2 = "scf.for"(%arg0, %arg1, %arg2, %arg6) ({
      ^bb0(%arg7: index, %arg8: memref<2 × f32>):
        %3 = "memref.alloc"(%arg5) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
        "scf.yield"(%0) : (memref<2 × f32>) -> ()
      }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
      "scf.yield"(%0) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "test.copy"(%1, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "partial_hoist_multiple_loop_dependency"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    %1 = "scf.for"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%arg5: index, %arg6: memref<2 × f32>):
      %2 = "scf.for"(%arg0, %arg1, %arg2, %arg6) ({
      ^bb0(%arg7: index, %arg8: memref<2 × f32>):
        %3 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
        "scf.yield"(%0) : (memref<2 × f32>) -> ()
      }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
      "scf.yield"(%0) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "test.copy"(%1, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "hoist_alloca"} : () -> ()
}) : () -> ()


