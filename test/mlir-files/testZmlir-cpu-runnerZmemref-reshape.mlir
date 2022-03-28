"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (memref<* × f32>) -> (), llvm.emit_c_interface, sym_name = "print_memref_f32", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × 3 × f32>
    %3 = "memref.dim"(%2, %0) : (memref<2 × 3 × f32>, index) -> index
    %4 = "memref.dim"(%2, %1) : (memref<2 × 3 × f32>, index) -> index
    "scf.parallel"(%0, %0, %3, %4, %1, %1) ({
    ^bb0(%arg0: index, %arg1: index):
      %9 = "arith.muli"(%arg0, %4) : (index, index) -> index
      %10 = "arith.addi"(%9, %arg1) : (index, index) -> index
      %11 = "arith.index_cast"(%10) : (index) -> i64
      %12 = "arith.sitofp"(%11) : (i64) -> f32
      "memref.store"(%12, %2, %arg0, %arg1) : (f32, memref<2 × 3 × f32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) {operand_segment_sizes = dense<[2, 2, 2, 0]> : vector<4 × i32>} : (index, index, index, index, index, index) -> ()
    %5 = "memref.cast"(%2) : (memref<2 × 3 × f32>) -> memref<* × f32>
    "func.call"(%5) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    %6 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × inde × >
    %7 = "arith.constant"() {value = 2 : index} : () -> index
    %8 = "arith.constant"() {value = 3 : index} : () -> index
    "memref.store"(%8, %6, %0) : (index, memref<2 × inde × >, index) -> ()
    "memref.store"(%7, %6, %1) : (index, memref<2 × inde × >, index) -> ()
    "func.call"(%2, %6) {callee = @reshape_ranked_memref_to_ranked} : (memref<2 × 3 × f32>, memref<2 × inde × >) -> ()
    "func.call"(%2, %6) {callee = @reshape_unranked_memref_to_ranked} : (memref<2 × 3 × f32>, memref<2 × inde × >) -> ()
    "func.call"(%2, %6) {callee = @reshape_ranked_memref_to_unranked} : (memref<2 × 3 × f32>, memref<2 × inde × >) -> ()
    "func.call"(%2, %6) {callee = @reshape_unranked_memref_to_unranked} : (memref<2 × 3 × f32>, memref<2 × inde × >) -> ()
    "memref.dealloc"(%2) : (memref<2 × 3 × f32>) -> ()
    "memref.dealloc"(%6) : (memref<2 × inde × >) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<2 × 3 × f32>, %arg1: memref<2 × inde × >):
    %0 = "memref.reshape"(%arg0, %arg1) : (memref<2 × 3 × f32>, memref<2 × inde × >) -> memref<? × ? × f32>
    %1 = "memref.cast"(%0) : (memref<? × ? × f32>) -> memref<* × f32>
    "func.call"(%1) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × 3 × f32>, memref<2 × inde × >) -> (), sym_name = "reshape_ranked_memref_to_ranked"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<2 × 3 × f32>, %arg1: memref<2 × inde × >):
    %0 = "memref.cast"(%arg0) : (memref<2 × 3 × f32>) -> memref<* × f32>
    %1 = "memref.reshape"(%arg0, %arg1) : (memref<2 × 3 × f32>, memref<2 × inde × >) -> memref<? × ? × f32>
    %2 = "memref.cast"(%1) : (memref<? × ? × f32>) -> memref<* × f32>
    "func.call"(%2) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × 3 × f32>, memref<2 × inde × >) -> (), sym_name = "reshape_unranked_memref_to_ranked"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<2 × 3 × f32>, %arg1: memref<2 × inde × >):
    %0 = "memref.cast"(%arg1) : (memref<2 × inde × >) -> memref<? × inde × >
    %1 = "memref.reshape"(%arg0, %0) : (memref<2 × 3 × f32>, memref<? × inde × >) -> memref<* × f32>
    "func.call"(%1) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × 3 × f32>, memref<2 × inde × >) -> (), sym_name = "reshape_ranked_memref_to_unranked"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<2 × 3 × f32>, %arg1: memref<2 × inde × >):
    %0 = "memref.cast"(%arg0) : (memref<2 × 3 × f32>) -> memref<* × f32>
    %1 = "memref.cast"(%arg1) : (memref<2 × inde × >) -> memref<? × inde × >
    %2 = "memref.reshape"(%arg0, %1) : (memref<2 × 3 × f32>, memref<? × inde × >) -> memref<* × f32>
    "func.call"(%2) {callee = @print_memref_f32} : (memref<* × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × 3 × f32>, memref<2 × inde × >) -> (), sym_name = "reshape_unranked_memref_to_unranked"} : () -> ()
}) : () -> ()


