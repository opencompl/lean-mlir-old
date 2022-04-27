"builtin.module"() ({
^bb0:
}) : () -> ()





































"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = -1 : index} : () -> index
    %2 = "arith.constant"() {value = 111 : index} : () -> index
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<9 × 9 × i32>
    %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<111 × i32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %5 = "affine.apply"(%arg0, %arg1) {map = #map0} : (index, index) -> index
        %6 = "affine.apply"(%arg0, %arg1) {map = #map1} : (index, index) -> index
        %7 = "affine.load"(%3, %5, %6) {map = #map2} : (memref<9 × 9 × i32>, index, index) -> i32
        %8 = "affine.apply"(%arg0, %arg1) {map = #map3} : (index, index) -> index
        %9 = "affine.load"(%4, %8) {map = #map4} : (memref<111 × i32>, index) -> i32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map5, step = 1 : index, upper_bound = #map6} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map5, step = 1 : index, upper_bound = #map6} : () -> ()
    "affine.for"() ({
    ^bb0(%arg0: index):
      %5 = "affine.load"(%4, %0) {map = #map4} : (memref<111 × i32>, index) -> i32
      %6 = "affine.load"(%4, %2) {map = #map4} : (memref<111 × i32>, index) -> i32
      "affine.store"(%6, %4, %1) {map = #map4} : (i32, memref<111 × i32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<128 × 64 × 64 × i32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map8} : (index, index, index) -> index
        %3 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map9} : (index, index, index) -> index
        %4 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map10} : (index, index, index) -> index
        %5 = "affine.load"(%1, %2, %3, %4) {map = #map11} : (memref<128 × 64 × 64 × i32>, index, index, index) -> i32
        %6 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map12} : (index, index, index) -> index
        %7 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map13} : (index, index, index) -> index
        %8 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map14} : (index, index, index) -> index
        "affine.store"(%5, %1, %6, %7, %8) {map = #map11} : (i32, memref<128 × 64 × 64 × i32>, index, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map7, step = 1 : index, upper_bound = #map15} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map15} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_mod_floordiv_ceildiv"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<257 × 256 × i32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<257 × i32>
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × i32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %4 = "affine.apply"(%arg0, %arg1) {map = #map16} : (index, index) -> index
        %5 = "affine.load"(%1, %4, %0) {map = #map2} : (memref<257 × 256 × i32>, index, index) -> i32
        %6 = "affine.apply"(%arg0, %arg0) {map = #map17} : (index, index) -> index
        %7 = "affine.load"(%3, %6) {map = #map4} : (memref<1 × i32>, index) -> i32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map7, step = 1 : index, upper_bound = #map15} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map15} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_no_out_of_bounds"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<128 × 64 × 64 × i32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %2 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map8} : (index, index, index) -> index
        %3 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map9} : (index, index, index) -> index
        %4 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map10} : (index, index, index) -> index
        %5 = "affine.load"(%1, %2, %3, %4) {map = #map11} : (memref<128 × 64 × 64 × i32>, index, index, index) -> i32
        %6 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map12} : (index, index, index) -> index
        %7 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map13} : (index, index, index) -> index
        %8 = "affine.apply"(%arg0, %arg1, %arg1) {map = #map14} : (index, index, index) -> index
        "affine.store"(%5, %1, %6, %7, %8) {map = #map11} : (i32, memref<128 × 64 × 64 × i32>, index, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map7, step = 1 : index, upper_bound = #map15} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map15} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "mod_div"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<256 × 256 × i32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %1 = "affine.apply"(%arg0, %arg1) {map = #map18} : (index, index) -> index
        %2 = "affine.apply"(%arg0, %arg1) {map = #map19} : (index, index) -> index
        %3 = "affine.load"(%0, %1, %2) {map = #map2} : (memref<256 × 256 × i32>, index, index) -> i32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map7, step = 1 : index, upper_bound = #map15} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map15} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "mod_floordiv_nested"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × i32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      %1 = "affine.apply"(%arg1, %arg0) {map = #map20} : (index, index) -> index
      %2 = "affine.load"(%0, %1) {map = #map4} : (memref<10 × i32>, index) -> i32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "test_semi_affine_bailout"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × 2 × i32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      %1 = "affine.apply"(%arg0) {map = #map21} : (index) -> index
      %2 = "affine.apply"(%arg0) {map = #map22} : (index) -> index
      %3 = "affine.load"(%0, %1, %2) {map = #map2} : (memref<2 × 2 × i32>, index, index) -> i32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map23} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "multi_mod_floordiv"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2x2x3x3x16x1xi32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 9 × i32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %3 = "affine.apply"(%arg0, %arg1) {map = #map24} : (index, index) -> index
        %4 = "affine.apply"(%3) {map = #map25} : (index) -> index
        %5 = "affine.apply"(%3) {map = #map26} : (index) -> index
        %6 = "affine.apply"(%3) {map = #map27} : (index) -> index
        %7 = "affine.apply"(%3) {map = #map22} : (index) -> index
        %8 = "affine.apply"(%3) {map = #map28} : (index) -> index
        %9 = "affine.apply"(%3) {map = #map29} : (index) -> index
        %10 = "affine.load"(%1, %4, %5, %7, %8, %6, %9) {map = #map30} : (memref<2x2x3x3x16x1xi32>, index, index, index, index, index, index) -> i32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map7, step = 1 : index, upper_bound = #map31} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map23} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "delinearize_mod_floordiv"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<i32>):
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    "affine.store"(%0, %arg0) {map = #map32} : (i32, memref<i32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<i32>) -> (), sym_name = "zero_d_memref"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × i32>
    %1 = "arith.constant"() {value = 9 : i32} : () -> i32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map33} : (index) -> index
      "affine.store"(%1, %0, %2) {map = #map4} : (i32, memref<1 × i32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map6, step = 1 : index, upper_bound = #map34} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "out_of_bounds"} : () -> ()
}) : () -> ()









"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<4 × 4 × 16 × 1 × f32>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1x2x3x3x16x1xf32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        %2 = "affine.apply"(%arg1, %arg2) {map = #map0} : (index, index) -> index
        %3 = "affine.apply"(%arg1, %arg2) {map = #map1} : (index, index) -> index
        %4 = "affine.apply"(%arg1, %arg2) {map = #map2} : (index, index) -> index
        %5 = "affine.load"(%arg0, %2, %0, %4) {map = #map3} : (memref<4 × 4 × 16 × 1 × f32>, index, index, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map4, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<4 × 4 × 16 × 1 × f32>) -> (), sym_name = "test_complex_mod_floordiv"} : () -> ()
}) : () -> ()







"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<7 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<6 × f32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"(%arg0, %arg0) ({
      ^bb0(%arg1: index):
        %2 = "affine.load"(%0, %arg1) {map = #map0} : (memref<7 × f32>, index) -> f32
        %3 = "affine.load"(%1, %arg1) {map = #map0} : (memref<6 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map3, step = 1 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_mod_bound"} : () -> ()
}) : () -> ()








"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1027 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1026 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<4096 × f32>
    %3 = "arith.constant"() {value = 2048 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"(%arg0, %arg0) ({
      ^bb0(%arg1: index):
        %4 = "affine.load"(%0, %arg1) {map = #map0} : (memref<1027 × f32>, index) -> f32
        %5 = "affine.load"(%1, %arg1) {map = #map0} : (memref<1026 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (index, index) -> ()
      "affine.for"(%3) ({
      ^bb0(%arg1: index):
        %4 = "affine.load"(%2, %arg1) {map = #map0} : (memref<4096 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map3, step = 1 : index, upper_bound = #map4} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map3, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_floordiv_bound"} : () -> ()
}) : () -> ()










"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<1024 × f32>):
    "affine.for"() ({
    ^bb0(%arg1: index):
      %0 = "affine.apply"(%arg1) {map = #map0} : (index) -> index
      "affine.for"(%0, %0) ({
      ^bb0(%arg2: index):
        %1 = "affine.load"(%arg0, %arg2) {map = #map1} : (memref<1024 × f32>, index) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map3, step = 4 : index, upper_bound = #map4} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<1024 × f32>) -> (), sym_name = "non_composed_bound_operand"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<f32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      %1 = "affine.load"(%0) {map = #map5} : (memref<f32>) -> f32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map6, step = 1 : index, upper_bound = #map7} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "zero_d_memref"} : () -> ()
}) : () -> ()


