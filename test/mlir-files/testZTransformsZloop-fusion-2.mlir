"builtin.module"() ({
^bb0:
}) : () -> ()









"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<64 × 4 × f32>, %arg1: memref<64 × 4 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 4 × f32>
    %1 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.for"() ({
      ^bb0(%arg3: index):
        "affine.store"(%1, %0, %arg2, %arg3) {map = #map0} : (f32, memref<64 × 4 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.for"() ({
      ^bb0(%arg3: index):
        "affine.for"() ({
        ^bb0(%arg4: index):
          %2 = "affine.load"(%arg1, %arg3, %arg4, %arg2) {map = #map4} : (memref<64 × 4 × f32>, index, index, index) -> f32
          "op0"(%2) : (f32) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
        "affine.for"() ({
        ^bb0(%arg4: index):
          "affine.for"() ({
          ^bb0(%arg5: index):
            %2 = "affine.load"(%arg0, %arg4, %arg5, %arg3) {map = #map4} : (memref<64 × 4 × f32>, index, index, index) -> f32
            "op1"(%2) : (f32) -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
          "affine.for"() ({
          ^bb0(%arg5: index):
            %2 = "op2"() : () -> f32
            %3 = "affine.load"(%0, %arg4, %arg5, %arg2) {map = #map6} : (memref<64 × 4 × f32>, index, index, index) -> f32
            %4 = "arith.addf"(%3, %2) : (f32, f32) -> f32
            "affine.store"(%4, %0, %arg4, %arg5, %arg2) {map = #map6} : (f32, memref<64 × 4 × f32>, index, index, index) -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<64 × 4 × f32>, memref<64 × 4 × f32>) -> (), sym_name = "should_fuse_at_depth_above_loop_carried_dependence"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_only_two_loops_and_remove_producer"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%1, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.load"(%0, %arg1) {map = #map0} : (memref<10 × f32>, index) -> f32
        "affine.store"(%2, %0, %arg1) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_after_one_loop_interchange"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<6 × 8 × f32>
    %1 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.store"(%1, %0, %arg0, %arg1) {map = #map0} : (f32, memref<6 × 8 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          "affine.for"() ({
          ^bb0(%arg3: index):
            %2 = "affine.load"(%0, %arg1, %arg3) {map = #map0} : (memref<6 × 8 × f32>, index, index) -> f32
            %3 = "arith.addf"(%2, %2) : (f32, f32) -> f32
            "affine.store"(%3, %0, %arg1, %arg3) {map = #map0} : (f32, memref<6 × 8 × f32>, index, index) -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map4} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_after_two_loop_interchanges"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10 × f32>):
    %0 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.store"(%0, %arg0, %arg1) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      %1 = "affine.load"(%arg0, %arg1) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%1, %arg0, %arg1) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"(%arg0) : (memref<10 × f32>) -> ()
  }) {function_type = (memref<10 × f32>) -> memref<10 × f32>, sym_name = "should_fuse_live_out_writer"} : () -> ()
}) : () -> ()









"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<128 × 8 × f32>, %arg1: memref<32 × 8 × f32>, %arg2: f32):
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        "affine.store"(%arg2, %arg1, %arg3, %arg4) {map = #map0} : (f32, memref<32 × 8 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        "affine.for"() ({
        ^bb0(%arg5: index):
          "affine.for"() ({
          ^bb0(%arg6: index):
            %0 = "affine.load"(%arg0, %arg5, %arg6, %arg4) {map = #map4} : (memref<128 × 8 × f32>, index, index, index) -> f32
            %1 = "foo"(%0) : (f32) -> f32
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
          "affine.for"() ({
          ^bb0(%arg6: index):
            %0 = "affine.load"(%arg1, %arg3, %arg6, %arg4) {map = #map4} : (memref<32 × 8 × f32>, index, index, index) -> f32
            %1 = "arith.addf"(%0, %0) : (f32, f32) -> f32
            "affine.store"(%1, %arg1, %arg3, %arg6, %arg4) {map = #map4} : (f32, memref<32 × 8 × f32>, index, index, index) -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map1, step = 1 : index, upper_bound = #map5} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"(%arg1) : (memref<32 × 8 × f32>) -> ()
  }) {function_type = (memref<128 × 8 × f32>, memref<32 × 8 × f32>, f32) -> memref<32 × 8 × f32>, sym_name = "slice_tile"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    %3 = "arith.constant"() {value = 0 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          %4 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
          %5 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
          %6 = "affine.apply"(%4, %5) {map = #map1} : (index, index) -> index
          "affine.store"(%2, %0, %6) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          %4 = "affine.load"(%0, %3) {map = #map0} : (memref<10 × f32>, index) -> f32
          "affine.yield"() : () -> ()
        }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_add_slice_bounds"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10 × 10 × f32>, %arg1: memref<10 × 10 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %1 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    %2 = "arith.constant"() {value = 1.00000000 : f32} : () -> f32
    %3 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.for"() ({
      ^bb0(%arg3: index):
        "affine.store"(%3, %0, %arg2, %arg3) {map = #map0} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.for"() ({
      ^bb0(%arg3: index):
        "affine.store"(%1, %arg0, %arg2, %arg3) {map = #map0} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.for"() ({
      ^bb0(%arg3: index):
        %4 = "affine.load"(%0, %arg2, %arg3) {map = #map0} : (memref<10 × 10 × f32>, index, index) -> f32
        %5 = "affine.load"(%arg0, %arg2, %arg3) {map = #map0} : (memref<10 × 10 × f32>, index, index) -> f32
        %6 = "arith.mulf"(%4, %5) : (f32, f32) -> f32
        "affine.store"(%6, %arg0, %arg2, %arg3) {map = #map0} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.for"() ({
      ^bb0(%arg3: index):
        "affine.store"(%2, %arg1, %arg2, %arg3) {map = #map0} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.for"() ({
      ^bb0(%arg3: index):
        %4 = "affine.load"(%0, %arg2, %arg3) {map = #map0} : (memref<10 × 10 × f32>, index, index) -> f32
        %5 = "affine.load"(%arg1, %arg2, %arg3) {map = #map0} : (memref<10 × 10 × f32>, index, index) -> f32
        %6 = "arith.addf"(%4, %5) : (f32, f32) -> f32
        "affine.store"(%6, %arg1, %arg2, %arg3) {map = #map0} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10 × 10 × f32>, memref<10 × 10 × f32>) -> (), sym_name = "should_fuse_init_loops_siblings_then_shared_producer"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × 10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %5 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.store"(%5, %0, %arg0, %arg1) {map = #map0} : (f32, memref<10 × 10 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %6 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<10 × 10 × f32>, index, index) -> f32
        %7 = "affine.load"(%1, %arg1) {map = #map3} : (memref<10 × f32>, index) -> f32
        %8 = "arith.mulf"(%6, %7) : (f32, f32) -> f32
        %9 = "affine.load"(%3, %arg1) {map = #map3} : (memref<10 × f32>, index) -> f32
        %10 = "arith.addf"(%8, %9) : (f32, f32) -> f32
        "affine.store"(%10, %3, %arg1) {map = #map3} : (f32, memref<10 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %6 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<10 × 10 × f32>, index, index) -> f32
        %7 = "affine.load"(%2, %arg1) {map = #map3} : (memref<10 × f32>, index) -> f32
        %8 = "arith.mulf"(%6, %7) : (f32, f32) -> f32
        %9 = "affine.load"(%4, %arg1) {map = #map3} : (memref<10 × f32>, index) -> f32
        %10 = "arith.addf"(%8, %9) : (f32, f32) -> f32
        "affine.store"(%10, %4, %arg1) {map = #map3} : (f32, memref<10 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "two_matrix_vector_products"} : () -> ()
}) : () -> ()






"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × 16 × f32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %1 = "op1"() : () -> f32
        "affine.store"(%1, %0, %arg0, %arg1) {map = #map0} : (f32, memref<100 × 16 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, slice_fusion_barrier = true, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %1 = "affine.load"(%0, %arg0, %arg1) {map = #map0} : (memref<100 × 16 × f32>, index, index) -> f32
        "op2"(%1) : (f32) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_not_slice_past_slice_barrier"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<4x4x16x1xf32>, %arg1: memref<144 × 9 × f32>, %arg2: memref<9 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<144 × 4 × f32>
    %1 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        "affine.for"() ({
        ^bb0(%arg5: index):
          %2 = "affine.apply"(%arg3, %arg5) {map = #map0} : (index, index) -> index
          "affine.store"(%1, %0, %2, %arg4) {map = #map1} : (f32, memref<144 × 4 × f32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : () -> ()
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        "affine.for"() ({
        ^bb0(%arg5: index):
          "affine.for"() ({
          ^bb0(%arg6: index):
            %2 = "affine.apply"(%arg3, %arg6) {map = #map0} : (index, index) -> index
            %3 = "affine.load"(%0, %2, %arg5) {map = #map1} : (memref<144 × 4 × f32>, index, index) -> f32
            "affine.yield"() : () -> ()
          }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<4x4x16x1xf32>, memref<144 × 9 × f32>, memref<9 × f32>) -> (), sym_name = "fuse_across_dim_mismatch"} : () -> ()
}) : () -> ()
















"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2x2x3x3x16x1xf32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 9 × f32>
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<144 × 4 × f32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        %4 = "affine.apply"(%arg1, %arg2) {map = #map0} : (index, index) -> index
        %5 = "affine.apply"(%arg1, %arg2) {map = #map1} : (index, index) -> index
        %6 = "affine.apply"(%arg1, %arg2) {map = #map2} : (index, index) -> index
        %7 = "affine.apply"(%arg1, %arg2) {map = #map3} : (index, index) -> index
        %8 = "affine.apply"(%arg1, %arg2) {map = #map4} : (index, index) -> index
        %9 = "affine.load"(%1, %4, %5, %7, %8, %6, %0) {map = #map5} : (memref<2x2x3x3x16x1xf32>, index, index, index, index, index, index) -> f32
        "affine.store"(%9, %2, %arg1, %arg2) {map = #map6} : (f32, memref<64 × 9 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map7, step = 1 : index, upper_bound = #map8} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map9} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        "affine.for"() ({
        ^bb0(%arg3: index):
          %4 = "affine.apply"(%arg2, %arg3) {map = #map10} : (index, index) -> index
          %5 = "affine.load"(%2, %4, %arg1) {map = #map6} : (memref<64 × 9 × f32>, index, index) -> f32
          "affine.yield"() : () -> ()
        }) {lower_bound = #map7, step = 1 : index, upper_bound = #map11} : () -> ()
        "affine.for"() ({
        ^bb0(%arg3: index):
          %4 = "affine.apply"(%arg1, %arg3) {map = #map10} : (index, index) -> index
          "affine.store"(%arg0, %3, %4, %arg2) {map = #map6} : (f32, memref<144 × 4 × f32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map7, step = 1 : index, upper_bound = #map11} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map7, step = 1 : index, upper_bound = #map12} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map8} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        "affine.for"() ({
        ^bb0(%arg3: index):
          "affine.for"() ({
          ^bb0(%arg4: index):
            %4 = "affine.apply"(%arg3, %arg4) {map = #map13} : (index, index) -> index
            %5 = "affine.load"(%2, %4, %arg2) {map = #map6} : (memref<64 × 9 × f32>, index, index) -> f32
            "affine.yield"() : () -> ()
          }) {lower_bound = #map7, step = 1 : index, upper_bound = #map11} : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map7, step = 1 : index, upper_bound = #map12} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map7, step = 1 : index, upper_bound = #map8} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map8} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (f32) -> (), sym_name = "fuse_across_varying_dims_complex"} : () -> ()
}) : () -> ()









"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<100 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<100 × f32>, index) -> f32
      "affine.for"() ({
      ^bb0(%arg1: index):
        %4 = "affine.load"(%0, %arg1) {map = #map0} : (memref<100 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map3, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map5, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_with_slice_union"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1024 × 1024 × f32>, %arg1: memref<1024 × 1024 × f32>, %arg2: memref<1024 × 1024 × f32>, %arg3: memref<1024 × 1024 × f32>):
    "affine.for"() ({
    ^bb0(%arg4: index):
      "affine.for"() ({
      ^bb0(%arg5: index):
        %0 = "affine.load"(%arg3, %arg4, %arg5) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
        %1 = "affine.load"(%arg2, %arg4, %arg5) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
        %2 = "arith.addf"(%1, %0) : (f32, f32) -> f32
        "affine.store"(%2, %arg2, %arg4, %arg5) {map = #map0} : (f32, memref<1024 × 1024 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg4: index):
      "affine.for"() ({
      ^bb0(%arg5: index):
        "affine.for"() ({
        ^bb0(%arg6: index):
          %0 = "affine.load"(%arg1, %arg6, %arg5) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %1 = "affine.load"(%arg0, %arg4, %arg6) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %2 = "arith.mulf"(%1, %0) : (f32, f32) -> f32
          %3 = "affine.load"(%arg2, %arg4, %arg5) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %4 = "arith.addf"(%3, %2) : (f32, f32) -> f32
          "affine.store"(%4, %arg2, %arg4, %arg5) {map = #map0} : (f32, memref<1024 × 1024 × f32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>) -> (), sym_name = "affine_add_mm_fused"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1024 × 1024 × f32>, %arg1: memref<1024 × 1024 × f32>, %arg2: memref<1024 × 1024 × f32>, %arg3: memref<1024 × 1024 × f32>, %arg4: memref<1024 × 1024 × f32>):
    %0 = "arith.constant"() {value = 0.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg5: index):
      "affine.for"() ({
      ^bb0(%arg6: index):
        "affine.store"(%0, %arg2, %arg5, %arg6) {map = #map0} : (f32, memref<1024 × 1024 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg5: index):
      "affine.for"() ({
      ^bb0(%arg6: index):
        "affine.store"(%0, %arg4, %arg5, %arg6) {map = #map0} : (f32, memref<1024 × 1024 × f32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg5: index):
      "affine.for"() ({
      ^bb0(%arg6: index):
        "affine.for"() ({
        ^bb0(%arg7: index):
          %1 = "affine.load"(%arg1, %arg7, %arg6) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %2 = "affine.load"(%arg0, %arg5, %arg7) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %3 = "arith.mulf"(%2, %1) : (f32, f32) -> f32
          %4 = "affine.load"(%arg2, %arg5, %arg6) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %5 = "arith.addf"(%4, %3) : (f32, f32) -> f32
          "affine.store"(%5, %arg2, %arg5, %arg6) {map = #map0} : (f32, memref<1024 × 1024 × f32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg5: index):
      "affine.for"() ({
      ^bb0(%arg6: index):
        "affine.for"() ({
        ^bb0(%arg7: index):
          %1 = "affine.load"(%arg1, %arg7, %arg6) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %2 = "affine.load"(%arg0, %arg5, %arg7) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %3 = "arith.mulf"(%2, %1) : (f32, f32) -> f32
          %4 = "affine.load"(%arg4, %arg5, %arg6) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %5 = "arith.addf"(%4, %3) : (f32, f32) -> f32
          "affine.store"(%5, %arg4, %arg5, %arg6) {map = #map0} : (f32, memref<1024 × 1024 × f32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>) -> (), sym_name = "affine_2mm_fused"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1024 × 1024 × f32>, %arg1: memref<1024 × 1024 × f32>, %arg2: memref<1024 × 1024 × f32>, %arg3: memref<1024 × 1024 × f32>, %arg4: memref<1024 × 1024 × f32>):
    "affine.for"() ({
    ^bb0(%arg5: index):
      "affine.for"() ({
      ^bb0(%arg6: index):
        "affine.for"() ({
        ^bb0(%arg7: index):
          %0 = "affine.load"(%arg1, %arg7, %arg6) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %1 = "affine.load"(%arg0, %arg5, %arg7) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %2 = "arith.mulf"(%1, %0) : (f32, f32) -> f32
          %3 = "affine.load"(%arg2, %arg5, %arg6) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %4 = "arith.addf"(%3, %2) : (f32, f32) -> f32
          "affine.store"(%4, %arg2, %arg5, %arg6) {map = #map0} : (f32, memref<1024 × 1024 × f32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg5: index):
      "affine.for"() ({
      ^bb0(%arg6: index):
        "affine.for"() ({
        ^bb0(%arg7: index):
          %0 = "affine.load"(%arg3, %arg7, %arg6) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %1 = "affine.load"(%arg2, %arg5, %arg7) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %2 = "arith.mulf"(%1, %0) : (f32, f32) -> f32
          %3 = "affine.load"(%arg4, %arg5, %arg6) {map = #map0} : (memref<1024 × 1024 × f32>, index, index) -> f32
          %4 = "arith.addf"(%3, %2) : (f32, f32) -> f32
          "affine.store"(%4, %arg4, %arg5, %arg6) {map = #map0} : (f32, memref<1024 × 1024 × f32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>, memref<1024 × 1024 × f32>) -> (), sym_name = "affine_2_dependent_mm_fused"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      %3 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.store"(%3, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_self_dependence_multi_store_producer"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %2 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.store"(%2, %1, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "should_fuse_dead_multi_store_producer"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<10 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 7.00000000 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.store"(%1, %arg0, %arg1) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.store"(%1, %0, %arg1) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "affine.for"() ({
    ^bb0(%arg1: index):
      %2 = "affine.load"(%0, %arg1) {map = #map0} : (memref<10 × f32>, index) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<10 × f32>) -> (), sym_name = "should_fuse_function_live_out_multi_store_producer"} : () -> ()
}) : () -> ()


