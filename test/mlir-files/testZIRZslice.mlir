


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    %0 = "memref.alloc"(%arg0, %arg2) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    %1 = "memref.alloc"(%arg2, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    %2 = "memref.alloc"(%arg0, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    %3 = "memref.alloc"(%arg0, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    "linalg.matmul"(%0, %1, %2) ({
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %4 = "arith.mulf"(%arg3, %arg4) : (f32, f32) -> f32
      %5 = "arith.addf"(%arg5, %4) : (f32, f32) -> f32
      "linalg.yield"(%5) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map0, #map1, #map2], operand_segment_sizes = dense<[2, 1]> : vector<2 × i32>} : (memref<? × ? × f32>, memref<? × ? × f32>, memref<? × ? × f32>) -> ()
    "linalg.matmul"(%0, %1, %3) ({
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %4 = "arith.mulf"(%arg3, %arg4) : (f32, f32) -> f32
      %5 = "arith.addf"(%arg5, %4) : (f32, f32) -> f32
      "linalg.yield"(%5) : (f32) -> ()
    }) {linalg.memoized_indexing_maps = [#map0, #map1, #map2], operand_segment_sizes = dense<[2, 1]> : vector<2 × i32>} : (memref<? × ? × f32>, memref<? × ? × f32>, memref<? × ? × f32>) -> ()
    "memref.dealloc"(%2) : (memref<? × ? × f32>) -> ()
    "memref.dealloc"(%1) : (memref<? × ? × f32>) -> ()
    "memref.dealloc"(%0) : (memref<? × ? × f32>) -> ()
    "memref.dealloc"(%3) : (memref<? × ? × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index) -> (), sym_name = "slicing_linalg_op"} : () -> ()
}) : () -> ()


