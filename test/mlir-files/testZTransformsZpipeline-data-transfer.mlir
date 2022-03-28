"builtin.module"() ({
^bb0:
}) : () -> ()


#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (32)>
#map3 = affine_map<() -> (8)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<256 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<32 × f32, 1>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × f32>
    %3 = "arith.constant"() {value = 0 : index} : () -> index
    %4 = "arith.constant"() {value = 32 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.dma_start"(%0, %arg0, %1, %arg0, %2, %3, %4) {dst_map = #map0, src_map = #map0, tag_map = #map0} : (memref<256 × f32>, index, memref<32 × f32, 1>, index, memref<1 × f32>, index, index) -> ()
      "affine.dma_wait"(%2, %3, %4) {tag_map = #map0} : (memref<1 × f32>, index, index) -> ()
      %5 = "affine.load"(%1, %arg0) {map = #map0} : (memref<32 × f32, 1>, index) -> f32
      %6 = "compute"(%5) : (f32) -> f32
      "affine.store"(%6, %1, %arg0) {map = #map0} : (f32, memref<32 × f32, 1>, index) -> ()
      "affine.for"() ({
      ^bb0(%arg1: index):
        "do_more_compute"(%arg0, %arg1) : (index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : () -> ()
    "memref.dealloc"(%2) : (memref<1 × f32>) -> ()
    "memref.dealloc"(%1) : (memref<32 × f32, 1>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "loop_nest_dma"} : () -> ()
}) : () -> ()


#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (512)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<512 × f32>, %arg1: memref<512 × f32>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 4 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg2: index):
      %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<4 × f32, 1>
      %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × i32>
      "affine.dma_start"(%arg0, %arg2, %2, %0, %3, %0, %1) {dst_map = #map0, src_map = #map0, tag_map = #map0} : (memref<512 × f32>, index, memref<4 × f32, 1>, index, memref<1 × i32>, index, index) -> ()
      "affine.dma_wait"(%3, %0, %1) {tag_map = #map0} : (memref<1 × i32>, index, index) -> ()
      "compute"(%arg2) : (index) -> ()
      "memref.dealloc"(%3) : (memref<1 × i32>) -> ()
      "memref.dealloc"(%2) : (memref<4 × f32, 1>) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 4 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<512 × f32>, memref<512 × f32>) -> (), sym_name = "loop_step"} : () -> ()
}) : () -> ()


#map0 = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d0 * 64 + d1 * 8)>
#map5 = affine_map<() -> (0)>
#map6 = affine_map<() -> (4)>
#map7 = affine_map<() -> (8)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<512 × 32 × vector<8 × f32>>, %arg1: memref<512 × 32 × vector<8 × f32>>, %arg2: memref<512 × 32 × vector<8 × f32>>):
    %0 = "arith.constant"() {value = 256 : index} : () -> index
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 4 × vector<8 × f32>, 2>
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 4 × vector<8 × f32>, 2>
    %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 4 × vector<8 × f32>, 2>
    %5 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × i32>
    %6 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × i32>
    %7 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × i32>
    "affine.for"() ({
    ^bb0(%arg3: index):
      %8 = "affine.apply"(%arg3) {map = #map0} : (index) -> index
      "affine.dma_start"(%arg2, %8, %1, %4, %1, %7, %1, %0) {dst_map = #map1, src_map = #map2, tag_map = #map3} : (memref<512 × 32 × vector<8 × f32>>, index, index, memref<64 × 4 × vector<8 × f32>, 2>, index, memref<2 × i32>, index, index) -> ()
      "affine.dma_wait"(%7, %1, %0) {tag_map = #map3} : (memref<2 × i32>, index, index) -> ()
      "affine.for"() ({
      ^bb0(%arg4: index):
        %9 = "affine.apply"(%arg3, %arg4) {map = #map4} : (index, index) -> index
        %10 = "affine.apply"(%arg4) {map = #map0} : (index) -> index
        "affine.dma_start"(%arg0, %9, %1, %2, %1, %5, %1, %0) {dst_map = #map1, src_map = #map2, tag_map = #map3} : (memref<512 × 32 × vector<8 × f32>>, index, index, memref<64 × 4 × vector<8 × f32>, 2>, index, memref<2 × i32>, index, index) -> ()
        "affine.dma_start"(%arg1, %10, %1, %3, %1, %6, %1, %0) {dst_map = #map1, src_map = #map2, tag_map = #map3} : (memref<512 × 32 × vector<8 × f32>>, index, index, memref<64 × 4 × vector<8 × f32>, 2>, index, memref<2 × i32>, index, index) -> ()
        "affine.dma_wait"(%5, %1, %0) {tag_map = #map3} : (memref<2 × i32>, index, index) -> ()
        "affine.dma_wait"(%6, %1, %0) {tag_map = #map3} : (memref<2 × i32>, index, index) -> ()
        "affine.for"() ({
        ^bb0(%arg5: index):
          "foo"() : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map5, step = 1 : index, upper_bound = #map6} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map5, step = 1 : index, upper_bound = #map7} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map5, step = 1 : index, upper_bound = #map7} : () -> ()
    "memref.dealloc"(%7) : (memref<2 × i32>) -> ()
    "memref.dealloc"(%6) : (memref<2 × i32>) -> ()
    "memref.dealloc"(%5) : (memref<2 × i32>) -> ()
    "memref.dealloc"(%4) : (memref<64 × 4 × vector<8 × f32>, 2>) -> ()
    "memref.dealloc"(%3) : (memref<64 × 4 × vector<8 × f32>, 2>) -> ()
    "memref.dealloc"(%2) : (memref<64 × 4 × vector<8 × f32>, 2>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<512 × 32 × vector<8 × f32>>, memref<512 × 32 × vector<8 × f32>>, memref<512 × 32 × vector<8 × f32>>) -> (), sym_name = "loop_dma_nested"} : () -> ()
}) : () -> ()


#map0 = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<() -> (8)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<512 × 32 × vector<8 × f32>>):
    %0 = "arith.constant"() {value = 256 : index} : () -> index
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 4 × vector<8 × f32>, 2>
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 4 × vector<8 × f32>, 2>
    %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<64 × 4 × vector<8 × f32>, 2>
    %5 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × i32>
    %6 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × i32>
    %7 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × i32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      %8 = "affine.apply"(%arg1) {map = #map0} : (index) -> index
      "affine.dma_start"(%arg0, %8, %1, %4, %1, %7, %1, %0) {dst_map = #map1, src_map = #map2, tag_map = #map3} : (memref<512 × 32 × vector<8 × f32>>, index, index, memref<64 × 4 × vector<8 × f32>, 2>, index, memref<2 × i32>, index, index) -> ()
      "affine.dma_wait"(%7, %1, %0) {tag_map = #map3} : (memref<2 × i32>, index, index) -> ()
      "affine.dma_start"(%4, %1, %arg0, %8, %1, %7, %1, %0) {dst_map = #map2, src_map = #map1, tag_map = #map3} : (memref<64 × 4 × vector<8 × f32>, 2>, index, memref<512 × 32 × vector<8 × f32>>, index, index, memref<2 × i32>, index, index) -> ()
      "affine.dma_wait"(%7, %1, %0) {tag_map = #map3} : (memref<2 × i32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
    "memref.dealloc"(%7) : (memref<2 × i32>) -> ()
    "memref.dealloc"(%6) : (memref<2 × i32>) -> ()
    "memref.dealloc"(%5) : (memref<2 × i32>) -> ()
    "memref.dealloc"(%4) : (memref<64 × 4 × vector<8 × f32>, 2>) -> ()
    "memref.dealloc"(%3) : (memref<64 × 4 × vector<8 × f32>, 2>) -> ()
    "memref.dealloc"(%2) : (memref<64 × 4 × vector<8 × f32>, 2>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<512 × 32 × vector<8 × f32>>) -> (), sym_name = "loop_dma_dependent"} : () -> ()
}) : () -> ()


#map0 = affine_map<(d0) -> (d0, d0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (16)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<512 × 32 × f32>):
    %0 = "arith.constant"() {value = 32 : index} : () -> index
    %1 = "arith.constant"() {value = 512 : index} : () -> index
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<32 × 32 × f32, 2>
    %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × i32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.dma_start"(%arg0, %2, %3, %2, %4, %2, %1) {dst_map = #map0, src_map = #map0, tag_map = #map1} : (memref<512 × 32 × f32>, index, memref<32 × 32 × f32, 2>, index, memref<1 × i32>, index, index) -> ()
      "affine.dma_wait"(%4, %2, %1) {tag_map = #map1} : (memref<1 × i32>, index, index) -> ()
      "foo"(%3) : (memref<32 × 32 × f32, 2>) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "memref.dealloc"(%4) : (memref<1 × i32>) -> ()
    "memref.dealloc"(%3) : (memref<32 × 32 × f32, 2>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<512 × 32 × f32>) -> (), sym_name = "escaping_use"} : () -> ()
}) : () -> ()


#map0 = affine_map<(d0) -> (d0, d0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (16)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<512 × 32 × f32>):
    %0 = "arith.constant"() {value = 32 : index} : () -> index
    %1 = "arith.constant"() {value = 512 : index} : () -> index
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<32 × 32 × f32, 2>
    %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × i32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.dma_start"(%arg0, %2, %3, %2, %4, %2, %1) {dst_map = #map0, src_map = #map0, tag_map = #map1} : (memref<512 × 32 × f32>, index, memref<32 × 32 × f32, 2>, index, memref<1 × i32>, index, index) -> ()
      "affine.dma_wait"(%4, %2, %1) {tag_map = #map1} : (memref<1 × i32>, index, index) -> ()
      "foo"(%4) : (memref<1 × i32>) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "memref.dealloc"(%4) : (memref<1 × i32>) -> ()
    "memref.dealloc"(%3) : (memref<32 × 32 × f32, 2>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<512 × 32 × f32>) -> (), sym_name = "escaping_tag"} : () -> ()
}) : () -> ()


#map0 = affine_map<(d0) -> (d0, d0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (16)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<512 × 32 × f32>):
    %0 = "arith.constant"() {value = 32 : index} : () -> index
    %1 = "arith.constant"() {value = 512 : index} : () -> index
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<32 × 32 × f32, 2>
    %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × i32>
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.dma_start"(%arg0, %2, %3, %2, %4, %2, %1) {dst_map = #map0, src_map = #map0, tag_map = #map1} : (memref<512 × 32 × f32>, index, memref<32 × 32 × f32, 2>, index, memref<1 × i32>, index, index) -> ()
      "affine.dma_wait"(%4, %2, %1) {tag_map = #map1} : (memref<1 × i32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    %5 = "affine.load"(%3, %2) {map = #map0} : (memref<32 × 32 × f32, 2>, index) -> f32
    "memref.dealloc"(%4) : (memref<1 × i32>) -> ()
    "memref.dealloc"(%3) : (memref<32 × 32 × f32, 2>) -> ()
    "func.return"(%5) : (f32) -> ()
  }) {function_type = (memref<512 × 32 × f32>) -> f32, sym_name = "live_out_use"} : () -> ()
}) : () -> ()


#map0 = affine_map<(d0) -> (d0, d0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (16)>
#map4 = affine_map<() -> (8)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<512 × 32 × f32>, %arg1: memref<? × ? × f32, 2>):
    %0 = "arith.constant"() {value = 512 : index} : () -> index
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × i32>
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.dma_start"(%arg0, %1, %arg1, %1, %2, %1, %0) {dst_map = #map0, src_map = #map0, tag_map = #map1} : (memref<512 × 32 × f32>, index, memref<? × ? × f32, 2>, index, memref<1 × i32>, index, index) -> ()
      "affine.dma_wait"(%2, %1, %0) {tag_map = #map1} : (memref<1 × i32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<512 × 32 × f32>, memref<? × ? × f32, 2>) -> (), sym_name = "dynamic_shape_dma_buffer"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<256 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<32 × f32, 1>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × f32>
    %3 = "arith.constant"() {value = 0 : index} : () -> index
    %4 = "arith.constant"() {value = 32 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.dma_start"(%0, %arg0, %1, %arg0, %2, %3, %4) {dst_map = #map1, src_map = #map1, tag_map = #map1} : (memref<256 × f32>, index, memref<32 × f32, 1>, index, memref<1 × f32>, index, index) -> ()
      "affine.dma_wait"(%2, %3, %4) {tag_map = #map1} : (memref<1 × f32>, index, index) -> ()
      "compute"(%1) : (memref<32 × f32, 1>) -> ()
      %5 = "affine.load"(%1, %arg0) {map = #map1} : (memref<32 × f32, 1>, index) -> f32
      "foo"(%5) : (f32) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
    "memref.dealloc"(%0) : (memref<256 × f32>) -> ()
    "memref.dealloc"(%1) : (memref<32 × f32, 1>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "escaping_and_indexed_use_mix"} : () -> ()
}) : () -> ()


