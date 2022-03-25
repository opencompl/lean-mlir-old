#map0 = affine_map<() -> (0)>
#map1 = affine_map<() -> (5)>
#map2 = affine_map<() -> (0, 0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<() -> (16)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<16x16xf32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<16x16xf32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<16x16xf32>
    %3 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    "linalg.fill"(%3, %0) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2 × i32>} : (f32, memref<16x16xf32>) -> ()
    "linalg.fill"(%3, %1) ({
    ^bb0(%arg0: f32, %arg1: f32):
      "linalg.yield"(%arg0) : (f32) -> ()
    }) {operand_segment_sizes = dense<1> : vector<2 × i32>} : (f32, memref<16x16xf32>) -> ()
    %4 = "arith.constant"() {value = 1 : index} : () -> index
    %5 = "func.call"() {callee = @rtclock} : () -> f64
    "affine.for"() ({
    ^bb0(%arg0: index):
      "linalg.fill"(%3, %2) ({
      ^bb0(%arg1: f32, %arg2: f32):
        "linalg.yield"(%arg1) : (f32) -> ()
      }) {operand_segment_sizes = dense<1> : vector<2 × i32>} : (f32, memref<16x16xf32>) -> ()
      "func.call"(%0, %1, %2) {callee = @sgemm_naive} : (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
    %6 = "func.call"() {callee = @rtclock} : () -> f64
    %7 = "arith.subf"(%6, %5) : (f64, f64) -> f64
    %8 = "affine.load"(%2) {map = #map2} : (memref<16x16xf32>) -> f32
    "vector.print"(%8) : (f32) -> ()
    %9 = "arith.constant"() {value = 0 : index} : () -> index
    %10 = "arith.constant"() {value = 1 : index} : () -> index
    %11 = "arith.constant"() {value = 2 : index} : () -> index
    %12 = "memref.dim"(%2, %9) : (memref<16x16xf32>, index) -> index
    %13 = "memref.dim"(%2, %10) : (memref<16x16xf32>, index) -> index
    %14 = "memref.dim"(%0, %10) : (memref<16x16xf32>, index) -> index
    %15 = "arith.muli"(%12, %13) : (index, index) -> index
    %16 = "arith.muli"(%15, %14) : (index, index) -> index
    %17 = "arith.muli"(%11, %16) : (index, index) -> index
    %18 = "arith.muli"(%4, %17) : (index, index) -> index
    %19 = "arith.index_cast"(%18) : (index) -> i16
    %20 = "arith.sitofp"(%19) : (i16) -> f64
    %21 = "arith.divf"(%20, %7) : (f64, f64) -> f64
    "func.call"(%21) {callee = @print_flops} : (f64) -> ()
    "memref.dealloc"(%0) : (memref<16x16xf32>) -> ()
    "memref.dealloc"(%1) : (memref<16x16xf32>) -> ()
    "memref.dealloc"(%2) : (memref<16x16xf32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg3: index):
      "affine.for"() ({
      ^bb0(%arg4: index):
        %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × f32>
        %2 = "affine.load"(%arg2, %arg3, %arg4) {map = #map3} : (memref<16x16xf32>, index, index) -> f32
        "affine.store"(%2, %1, %0) {map = #map4} : (f32, memref<1 × f32>, index) -> ()
        "affine.for"() ({
        ^bb0(%arg5: index):
          %4 = "affine.load"(%arg0, %arg3, %arg5) {map = #map3} : (memref<16x16xf32>, index, index) -> f32
          %5 = "affine.load"(%arg1, %arg5, %arg4) {map = #map3} : (memref<16x16xf32>, index, index) -> f32
          %6 = "affine.load"(%1) {map = #map0} : (memref<1 × f32>) -> f32
          %7 = "arith.mulf"(%4, %5) : (f32, f32) -> f32
          %8 = "arith.addf"(%7, %6) : (f32, f32) -> f32
          "affine.store"(%8, %1) {map = #map0} : (f32, memref<1 × f32>) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map0, step = 1 : index, upper_bound = #map5} : () -> ()
        %3 = "affine.load"(%1, %0) {map = #map4} : (memref<1 × f32>, index) -> f32
        "affine.store"(%3, %arg2, %arg3, %arg4) {map = #map3} : (f32, memref<16x16xf32>, index, index) -> ()
        "memref.dealloc"(%1) : (memref<1 × f32>) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map0, step = 1 : index, upper_bound = #map5} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map0, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> (), sym_name = "sgemm_naive"} : () -> ()
  "func.func"() ({
  }) {function_type = (f64) -> (), sym_name = "print_flops", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> f64, sym_name = "rtclock", sym_visibility = "private"} : () -> ()
}) : () -> ()

// -----
