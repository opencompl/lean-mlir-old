#map = affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1024 × 64 × f32, 1>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "memref.alloc"(%1, %2) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32, 1>
    %4 = "memref.alloc"(%1) {operand_segment_sizes = dense<[0, 1]> : vector<2 × i32>} : (index) -> memref<2 × 4 × f32, #map, 1>
    %5 = "memref.alloc"(%2, %1) {operand_segment_sizes = dense<1> : vector<2 × i32>} : (index, index) -> memref<2 × ? × f32, #map, 1>
    %6 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × i32>
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "alloc"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1024 × 64 × f32, 1>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "memref.alloca"(%1, %2) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32, 1>
    %4 = "memref.alloca"(%1) {operand_segment_sizes = dense<[0, 1]> : vector<2 × i32>} : (index) -> memref<2 × 4 × f32, #map, 1>
    %5 = "memref.alloca"(%2, %1) {operand_segment_sizes = dense<1> : vector<2 × i32>} : (index, index) -> memref<2 × ? × f32, #map, 1>
    %6 = "memref.alloca"() {alignment = 64 : i64, operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × i32>
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "alloca"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1024 × 64 × f32>
    "memref.dealloc"(%0) : (memref<1024 × 64 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dealloc"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1024 × 64 × f32, 1>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "memref.load"(%0, %1, %2) : (memref<1024 × 64 × f32, 1>, index, index) -> f32
    "memref.store"(%3, %0, %1, %2) : (f32, memref<1024 × 64 × f32, 1>, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "load_store"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 32 : index} : () -> index
    %2 = "arith.constant"() {value = 16 : index} : () -> index
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<256 × f32>
    %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<256 × f32, 1>
    %5 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<1 × f32>
    %6 = "arith.constant"() {value = 256 : index} : () -> index
    "memref.dma_start"(%3, %0, %4, %0, %6, %5, %0) : (memref<256 × f32>, index, memref<256 × f32, 1>, index, index, memref<1 × f32>, index) -> ()
    "memref.dma_wait"(%5, %0, %6) : (memref<1 × f32>, index, index) -> ()
    "memref.dma_start"(%3, %0, %4, %0, %6, %5, %0, %1, %2) : (memref<256 × f32>, index, memref<256 × f32, 1>, index, index, memref<1 × f32>, index, index, index) -> ()
    "memref.dma_wait"(%5, %0, %6) : (memref<1 × f32>, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dma_ops"} : () -> ()
}) : () -> ()


