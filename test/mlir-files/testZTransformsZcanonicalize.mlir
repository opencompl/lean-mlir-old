








"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.subi"(%arg0, %arg0) : (i32, i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "test_subi_zero"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i32>):
    %0 = "arith.subi"(%arg0, %arg0) : (vector<4 × i32>, vector<4 × i32>) -> vector<4 × i32>
    "func.return"(%0) : (vector<4 × i32>) -> ()
  }) {function_type = (vector<4 × i32>) -> vector<4 × i32>, sym_name = "test_subi_zero_vector"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.subi"(%arg0, %arg0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%0) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "test_subi_zero_tensor"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<8 × 4 × f32>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "tensor.dim"(%arg0, %0) : (tensor<8 × 4 × f32>, index) -> index
    "func.return"(%1) : (index) -> ()
  }) {function_type = (tensor<8 × 4 × f32>) -> index, sym_name = "dim"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 42 : i32} : () -> i32
    %1 = "arith.addi"(%0, %arg0) : (i32, i32) -> i32
    %2 = "arith.subi"(%0, %arg0) : (i32, i32) -> i32
    "func.return"(%1, %2) : (i32, i32) -> ()
  }) {function_type = (i32) -> (i32, i32), sym_name = "test_commutative"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<8 × 4 × f32>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "tensor.dim"(%arg0, %0) : (tensor<8 × 4 × f32>, index) -> index
    "func.return"() : () -> ()
  }) {function_type = (tensor<8 × 4 × f32>) -> (), sym_name = "trivial_dce"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() {value = 4 : index} : () -> index
    %1 = "memref.alloc"(%0) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
    %2 = "memref.load"(%1, %arg0) : (memref<? × f32>, index) -> f32
    "memref.dealloc"(%1) : (memref<? × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "load_dce"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.addi"(%0, %arg0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "addi_zero"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.addi"(%0, %arg0) : (index, index) -> index
    "func.return"(%1) : (index) -> ()
  }) {function_type = (index) -> index, sym_name = "addi_zero_index"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i32>):
    %0 = "arith.constant"() {value = dense<0> : vector<4 × i32>} : () -> vector<4 × i32>
    %1 = "arith.addi"(%0, %arg0) : (vector<4 × i32>, vector<4 × i32>) -> vector<4 × i32>
    "func.return"(%1) : (vector<4 × i32>) -> ()
  }) {function_type = (vector<4 × i32>) -> vector<4 × i32>, sym_name = "addi_zero_vector"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.constant"() {value = dense<0> : tensor<4 × 5 × i32>} : () -> tensor<4 × 5 × i32>
    %1 = "arith.addi"(%arg0, %0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%1) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "addi_zero_tensor"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.muli"(%0, %arg0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "muli_zero"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.muli"(%0, %arg0) : (index, index) -> index
    "func.return"(%1) : (index) -> ()
  }) {function_type = (index) -> index, sym_name = "muli_zero_index"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i32>):
    %0 = "arith.constant"() {value = dense<0> : vector<4 × i32>} : () -> vector<4 × i32>
    %1 = "arith.muli"(%0, %arg0) : (vector<4 × i32>, vector<4 × i32>) -> vector<4 × i32>
    "func.return"(%1) : (vector<4 × i32>) -> ()
  }) {function_type = (vector<4 × i32>) -> vector<4 × i32>, sym_name = "muli_zero_vector"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.constant"() {value = dense<0> : tensor<4 × 5 × i32>} : () -> tensor<4 × 5 × i32>
    %1 = "arith.muli"(%arg0, %0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%1) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "muli_zero_tensor"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.muli"(%0, %arg0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "muli_one"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.muli"(%0, %arg0) : (index, index) -> index
    "func.return"(%1) : (index) -> ()
  }) {function_type = (index) -> index, sym_name = "muli_one_index"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i32>):
    %0 = "arith.constant"() {value = dense<1> : vector<4 × i32>} : () -> vector<4 × i32>
    %1 = "arith.muli"(%0, %arg0) : (vector<4 × i32>, vector<4 × i32>) -> vector<4 × i32>
    "func.return"(%1) : (vector<4 × i32>) -> ()
  }) {function_type = (vector<4 × i32>) -> vector<4 × i32>, sym_name = "muli_one_vector"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.constant"() {value = dense<1> : tensor<4 × 5 × i32>} : () -> tensor<4 × 5 × i32>
    %1 = "arith.muli"(%arg0, %0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%1) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "muli_one_tensor"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.andi"(%arg0, %arg0) : (i32, i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "and_self"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i32>):
    %0 = "arith.andi"(%arg0, %arg0) : (vector<4 × i32>, vector<4 × i32>) -> vector<4 × i32>
    "func.return"(%0) : (vector<4 × i32>) -> ()
  }) {function_type = (vector<4 × i32>) -> vector<4 × i32>, sym_name = "and_self_vector"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.andi"(%arg0, %arg0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%0) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "and_self_tensor"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.andi"(%arg0, %0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "and_zero"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.andi"(%arg0, %0) : (index, index) -> index
    "func.return"(%1) : (index) -> ()
  }) {function_type = (index) -> index, sym_name = "and_zero_index"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i32>):
    %0 = "arith.constant"() {value = dense<0> : vector<4 × i32>} : () -> vector<4 × i32>
    %1 = "arith.andi"(%arg0, %0) : (vector<4 × i32>, vector<4 × i32>) -> vector<4 × i32>
    "func.return"(%1) : (vector<4 × i32>) -> ()
  }) {function_type = (vector<4 × i32>) -> vector<4 × i32>, sym_name = "and_zero_vector"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.constant"() {value = dense<0> : tensor<4 × 5 × i32>} : () -> tensor<4 × 5 × i32>
    %1 = "arith.andi"(%arg0, %0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%1) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "and_zero_tensor"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.ori"(%arg0, %arg0) : (i32, i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "or_self"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i32>):
    %0 = "arith.ori"(%arg0, %arg0) : (vector<4 × i32>, vector<4 × i32>) -> vector<4 × i32>
    "func.return"(%0) : (vector<4 × i32>) -> ()
  }) {function_type = (vector<4 × i32>) -> vector<4 × i32>, sym_name = "or_self_vector"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.ori"(%arg0, %arg0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%0) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "or_self_tensor"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.ori"(%arg0, %0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "or_zero"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.ori"(%arg0, %0) : (index, index) -> index
    "func.return"(%1) : (index) -> ()
  }) {function_type = (index) -> index, sym_name = "or_zero_index"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i32>):
    %0 = "arith.constant"() {value = dense<0> : vector<4 × i32>} : () -> vector<4 × i32>
    %1 = "arith.ori"(%arg0, %0) : (vector<4 × i32>, vector<4 × i32>) -> vector<4 × i32>
    "func.return"(%1) : (vector<4 × i32>) -> ()
  }) {function_type = (vector<4 × i32>) -> vector<4 × i32>, sym_name = "or_zero_vector"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.constant"() {value = dense<0> : tensor<4 × 5 × i32>} : () -> tensor<4 × 5 × i32>
    %1 = "arith.ori"(%arg0, %0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%1) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "or_zero_tensor"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i4):
    %0 = "arith.constant"() {value = true} : () -> i1
    %1 = "arith.constant"() {value = -1 : i4} : () -> i4
    %2 = "arith.ori"(%arg0, %0) : (i1, i1) -> i1
    %3 = "arith.ori"(%arg1, %1) : (i4, i4) -> i4
    "func.return"(%2, %3) : (i1, i4) -> ()
  }) {function_type = (i1, i4) -> (i1, i4), sym_name = "or_all_ones"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.xori"(%arg0, %arg0) : (i32, i32) -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "xor_self"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i32>):
    %0 = "arith.xori"(%arg0, %arg0) : (vector<4 × i32>, vector<4 × i32>) -> vector<4 × i32>
    "func.return"(%0) : (vector<4 × i32>) -> ()
  }) {function_type = (vector<4 × i32>) -> vector<4 × i32>, sym_name = "xor_self_vector"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.xori"(%arg0, %arg0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%0) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "xor_self_tensor"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<4 × f32>, %arg1: f32):
    %0 = "memref.cast"(%arg0) : (memref<4 × f32>) -> memref<? × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "memref.dim"(%0, %1) : (memref<? × f32>, index) -> index
    %3 = "affine.load"(%0, %2) {map = #map0} : (memref<? × f32>, index) -> f32
    "memref.store"(%arg1, %0, %1) : (f32, memref<? × f32>, index) -> ()
    %4 = "memref.load"(%0, %1) : (memref<? × f32>, index) -> f32
    "memref.dealloc"(%0) : (memref<? × f32>) -> ()
    "func.return"(%3, %4) : (f32, f32) -> ()
  }) {function_type = (memref<4 × f32>, f32) -> (f32, f32), sym_name = "memref_cast_folding"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<42 × 42 × f64>):
    %0 = "memref.cast"(%arg0) : (memref<42 × 42 × f64>) -> memref<? × 42 × f64>
    %1 = "memref.cast"(%0) : (memref<? × 42 × f64>) -> memref<? × ? × f64>
    "test.user"(%1) : (memref<? × ? × f64>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<42 × 42 × f64>) -> (), sym_name = "fold_memref_cast_in_memref_cast"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<42 × 42 × f64>):
    %0 = "memref.cast"(%arg0) : (memref<42 × 42 × f64>) -> memref<? × 42 × f64>
    %1 = "memref.cast"(%0) : (memref<? × 42 × f64>) -> memref<42 × 42 × f64>
    "test.user"(%1) : (memref<42 × 42 × f64>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<42 × 42 × f64>) -> (), sym_name = "fold_memref_cast_chain"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 4 : index} : () -> index
    %1 = "memref.alloc"(%0) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dead_alloc_fold"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<4 × f32>
    "memref.dealloc"(%0) : (memref<4 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dead_dealloc_fold"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<4 × f32>
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  
    "memref.dealloc"(%0) : (memref<4 × f32>) -> ()
    "func.return"() : () -> ()
  ^bb2:  
    "memref.dealloc"(%0) : (memref<4 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1) -> (), sym_name = "dead_dealloc_fold_multi_use"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 4 : index} : () -> index
    %2 = "memref.alloc"(%1) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
    "memref.store"(%arg0, %2, %0) : (f32, memref<? × f32>, index) -> ()
    "memref.dealloc"(%2) : (memref<? × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (f32) -> (), sym_name = "write_only_alloc_fold"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 4 : index} : () -> index
    %2 = "memref.alloca"(%1) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × f32>
    "memref.store"(%arg0, %2, %0) : (f32, memref<? × f32>, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (f32) -> (), sym_name = "write_only_alloca_fold"} : () -> ()
  "func.func"() ({
    "func.func"() ({
      "func.return"() : () -> ()
    ^bb1:  
      "func.return"() : () -> ()
    }) {function_type = () -> (), sym_name = "nested"} : () -> ()
    "func.return"() : () -> ()
  ^bb1:  
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dead_block_elim"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 9 : index} : () -> index
    %2 = "arith.constant"() {value = 1024 : index} : () -> index
    %3 = "arith.constant"() {value = 512 : index} : () -> index
    %4 = "memref.alloc"(%arg0, %2) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    %5 = "memref.alloc"(%2, %3, %arg1) {operand_segment_sizes = dense<[3, 0]> : vector<2 × i32>} : (index, index, index) -> memref<4 × ? × 8 × ? × ? × f32>
    %6 = "memref.alloc"(%3, %2) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × i32>
    %7 = "memref.alloc"(%1, %1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    %8 = "memref.alloca"(%2, %3, %arg1) {operand_segment_sizes = dense<[3, 0]> : vector<2 × i32>} : (index, index, index) -> memref<4 × ? × 8 × ? × ? × f32>
    "affine.for"(%arg0) ({
    ^bb0(%arg2: index):
      "affine.for"() ({
      ^bb0(%arg3: index):
        %9 = "memref.load"(%4, %arg2, %arg3) : (memref<? × ? × f32>, index, index) -> f32
        "memref.store"(%9, %5, %0, %0, %arg2, %arg3, %0) : (f32, memref<4 × ? × 8 × ? × ? × f32>, index, index, index, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : (index) -> ()
    "func.return"(%5, %6, %7, %8) : (memref<4 × ? × 8 × ? × ? × f32>, memref<? × ? × i32>, memref<? × ? × f32>, memref<4 × ? × 8 × ? × ? × f32>) -> ()
  }) {function_type = (index, index) -> (memref<4 × ? × 8 × ? × ? × f32>, memref<? × ? × i32>, memref<? × ? × f32>, memref<4 × ? × 8 × ? × ? × f32>), sym_name = "dyn_shape_fold"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<? × i8>, %arg4: index, %arg5: index, %arg6: index):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 2 : index} : () -> index
    %3 = "memref.alloc"(%arg0, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    %4 = "memref.alloc"(%arg1, %arg2) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × 8 × ? × f32>
    %5 = "memref.dim"(%4, %2) : (memref<? × 8 × ? × f32>, index) -> index
    "affine.for"(%5) ({
    ^bb0(%arg7: index):
      %12 = "memref.alloc"(%arg0) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × i8>
      %13 = "memref.dim"(%12, %0) : (memref<? × i8>, index) -> index
      "affine.for"(%13) ({
      ^bb0(%arg8: index):
        %14 = "memref.dim"(%3, %0) : (memref<? × ? × f32>, index) -> index
        %15 = "memref.view"(%12, %0, %arg8, %14) : (memref<? × i8>, index, index, index) -> memref<? × ? × f32>
        %16 = "memref.subview"(%3, %0, %0, %14, %arg8, %1, %1) {operand_segment_sizes = dense<[1, 2, 2, 2]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808], static_sizes = [-1, -1], static_strides = [-9223372036854775808, -9223372036854775808]} : (memref<? × ? × f32>, index, index, index, index, index, index) -> memref<? × ? × f32, #map4>
        %17 = "memref.dim"(%15, %1) : (memref<? × ? × f32>, index) -> index
        %18 = "memref.dim"(%16, %0) : (memref<? × ? × f32, #map4>, index) -> index
        "affine.for"(%17, %18) ({
        ^bb0(%arg9: index):
          "foo"() : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map3, step = 1 : index, upper_bound = #map3} : (index, index) -> ()
        %19 = "memref.subview"(%3, %arg8) {operand_segment_sizes = dense<[1, 0, 1, 0]> : vector<4 × i32>, static_offsets = [0, 0], static_sizes = [17, -1], static_strides = [1, 1]} : (memref<? × ? × f32>, index) -> memref<17 × ? × f32, #map5>
        %20 = "memref.dim"(%15, %1) : (memref<? × ? × f32>, index) -> index
        %21 = "memref.dim"(%19, %1) : (memref<17 × ? × f32, #map5>, index) -> index
        "scf.for"(%20, %21, %1) ({
        ^bb0(%arg9: index):
          "foo"() : () -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map3} : (index) -> ()
    %6 = "memref.view"(%arg3, %0, %arg4, %arg6) : (memref<? × i8>, index, index, index) -> memref<? × ? × f32>
    %7 = "memref.view"(%arg3, %0, %arg6, %arg5) : (memref<? × i8>, index, index, index) -> memref<? × ? × f32>
    %8 = "memref.view"(%arg3, %0, %arg4, %arg5) : (memref<? × i8>, index, index, index) -> memref<? × ? × f32>
    %9 = "memref.dim"(%6, %0) : (memref<? × ? × f32>, index) -> index
    %10 = "memref.dim"(%6, %1) : (memref<? × ? × f32>, index) -> index
    %11 = "memref.dim"(%8, %1) : (memref<? × ? × f32>, index) -> index
    "scf.for"(%0, %9, %1) ({
    ^bb0(%arg7: index):
      "scf.for"(%0, %11, %1) ({
      ^bb0(%arg8: index):
        "scf.for"(%0, %10, %1) ({
        ^bb0(%arg9: index):
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<? × i8>, index, index, index) -> (), sym_name = "dim_op_fold"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 42 : index} : () -> index
    %1 = "arith.constant"() {value = 42 : index} : () -> index
    "func.return"(%0, %1) : (index, index) -> ()
  }) {function_type = () -> (index, index), sym_name = "merge_constants"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<8 × i32>):
    "affine.for"() ({
    ^bb0(%arg1: index):
      %0 = "arith.constant"() {value = 42 : i32} : () -> i32
      "memref.store"(%0, %arg0, %arg1) : (i32, memref<8 × i32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map6} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<8 × i32>) -> (), sym_name = "hoist_constant"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 512 : index} : () -> index
    %1 = "affine.apply"(%0) {map = #map7} : (index) -> index
    %2 = "affine.apply"(%0) {map = #map8} : (index) -> index
    %3 = "memref.alloc"(%1, %2) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    "func.return"(%3) : (memref<? × ? × f32>) -> ()
  }) {function_type = () -> memref<? × ? × f32>, sym_name = "const_fold_propagate"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "indirect_target"} : () -> ()
  "func.func"() ({
    %0 = "func.constant"() {value = @indirect_target} : () -> (() -> ())
    "func.call_indirect"(%0) : (() -> ()) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "indirect_call_folding"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = -43 : index} : () -> index
    %1 = "arith.constant"() {value = 42 : index} : () -> index
    %2 = "arith.remsi"(%0, %1) : (index, index) -> index
    %3 = "arith.constant"() {value = 0 : index} : () -> index
    %4 = "arith.cmpi"(%2, %3) {predicate = 2 : i64} : (index, index) -> i1
    %5 = "arith.addi"(%2, %1) : (index, index) -> index
    %6 = "arith.select"(%4, %5, %2) : (i1, index, index) -> index
    %7 = "arith.constant"() {value = 43 : index} : () -> index
    %8 = "arith.constant"() {value = 42 : index} : () -> index
    %9 = "arith.remsi"(%7, %8) : (index, index) -> index
    %10 = "arith.constant"() {value = 0 : index} : () -> index
    %11 = "arith.cmpi"(%9, %10) {predicate = 2 : i64} : (index, index) -> i1
    %12 = "arith.addi"(%9, %8) : (index, index) -> index
    %13 = "arith.select"(%11, %12, %9) : (i1, index, index) -> index
    "func.return"(%6, %13) : (index, index) -> ()
  }) {function_type = () -> (index, index), sym_name = "lowered_affine_mod"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = -43 : index} : () -> index
    %1 = "arith.constant"() {value = 42 : index} : () -> index
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    %3 = "arith.constant"() {value = -1 : index} : () -> index
    %4 = "arith.cmpi"(%0, %2) {predicate = 2 : i64} : (index, index) -> i1
    %5 = "arith.subi"(%3, %0) : (index, index) -> index
    %6 = "arith.select"(%4, %5, %0) : (i1, index, index) -> index
    %7 = "arith.divsi"(%6, %1) : (index, index) -> index
    %8 = "arith.subi"(%3, %7) : (index, index) -> index
    %9 = "arith.select"(%4, %8, %7) : (i1, index, index) -> index
    %10 = "arith.constant"() {value = 43 : index} : () -> index
    %11 = "arith.constant"() {value = 42 : index} : () -> index
    %12 = "arith.constant"() {value = 0 : index} : () -> index
    %13 = "arith.constant"() {value = -1 : index} : () -> index
    %14 = "arith.cmpi"(%10, %12) {predicate = 2 : i64} : (index, index) -> i1
    %15 = "arith.subi"(%13, %10) : (index, index) -> index
    %16 = "arith.select"(%14, %15, %10) : (i1, index, index) -> index
    %17 = "arith.divsi"(%16, %11) : (index, index) -> index
    %18 = "arith.subi"(%13, %17) : (index, index) -> index
    %19 = "arith.select"(%14, %18, %17) : (i1, index, index) -> index
    "func.return"(%9, %19) : (index, index) -> ()
  }) {function_type = () -> (index, index), sym_name = "lowered_affine_floordiv"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = -43 : index} : () -> index
    %1 = "arith.constant"() {value = 42 : index} : () -> index
    %2 = "arith.constant"() {value = 0 : index} : () -> index
    %3 = "arith.constant"() {value = 1 : index} : () -> index
    %4 = "arith.cmpi"(%0, %2) {predicate = 3 : i64} : (index, index) -> i1
    %5 = "arith.subi"(%2, %0) : (index, index) -> index
    %6 = "arith.subi"(%0, %3) : (index, index) -> index
    %7 = "arith.select"(%4, %5, %6) : (i1, index, index) -> index
    %8 = "arith.divsi"(%7, %1) : (index, index) -> index
    %9 = "arith.subi"(%2, %8) : (index, index) -> index
    %10 = "arith.addi"(%8, %3) : (index, index) -> index
    %11 = "arith.select"(%4, %9, %10) : (i1, index, index) -> index
    %12 = "arith.constant"() {value = 43 : index} : () -> index
    %13 = "arith.constant"() {value = 42 : index} : () -> index
    %14 = "arith.constant"() {value = 0 : index} : () -> index
    %15 = "arith.constant"() {value = 1 : index} : () -> index
    %16 = "arith.cmpi"(%12, %14) {predicate = 3 : i64} : (index, index) -> i1
    %17 = "arith.subi"(%14, %12) : (index, index) -> index
    %18 = "arith.subi"(%12, %15) : (index, index) -> index
    %19 = "arith.select"(%16, %17, %18) : (i1, index, index) -> index
    %20 = "arith.divsi"(%19, %13) : (index, index) -> index
    %21 = "arith.subi"(%14, %20) : (index, index) -> index
    %22 = "arith.addi"(%20, %15) : (index, index) -> index
    %23 = "arith.select"(%16, %21, %22) : (i1, index, index) -> index
    "func.return"(%11, %23) : (index, index) -> ()
  }) {function_type = () -> (index, index), sym_name = "lowered_affine_ceildiv"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<? × i32>):
    %0 = "memref.cast"(%arg0) : (memref<? × i32>) -> memref<? × i32>
    %1 = "memref.cast"(%0) : (memref<? × i32>) -> memref<2 × i32>
    %2 = "memref.cast"(%1) : (memref<2 × i32>) -> memref<2 × i32>
    "func.return"(%2) : (memref<2 × i32>) -> ()
  }) {function_type = (memref<? × i32>) -> memref<2 × i32>, sym_name = "cast_values"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2048 × i8>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 7 : index} : () -> index
    %3 = "arith.constant"() {value = 11 : index} : () -> index
    %4 = "arith.constant"() {value = 15 : index} : () -> index
    %5 = "memref.view"(%0, %4, %2, %3) : (memref<2048 × i8>, index, index, index) -> memref<? × ? × f32>
    %6 = "memref.load"(%5, %1, %1) : (memref<? × ? × f32>, index, index) -> f32
    %7 = "memref.view"(%0, %4, %arg0, %arg0, %2) : (memref<2048 × i8>, index, index, index, index) -> memref<? × ? × ? × f32>
    %8 = "memref.load"(%7, %1, %1, %1) : (memref<? × ? × ? × f32>, index, index, index) -> f32
    %9 = "memref.view"(%0, %4, %2) : (memref<2048 × i8>, index, index) -> memref<? × 4 × f32>
    %10 = "memref.load"(%9, %1, %1) : (memref<? × 4 × f32>, index, index) -> f32
    %11 = "memref.cast"(%0) : (memref<2048 × i8>) -> memref<? × i8>
    %12 = "memref.view"(%11, %4, %4, %2) : (memref<? × i8>, index, index, index) -> memref<? × ? × f32>
    %13 = "memref.load"(%12, %1, %1) : (memref<? × ? × f32>, index, index) -> f32
    "func.return"(%6, %8, %10, %13) : (f32, f32, f32, f32) -> ()
  }) {function_type = (index) -> (f32, f32, f32, f32), sym_name = "view"} : () -> ()
}) : () -> ()









"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 2 : index} : () -> index
    %3 = "arith.constant"() {value = 7 : index} : () -> index
    %4 = "arith.constant"() {value = 11 : index} : () -> index
    %5 = "arith.constant"() {value = 15 : index} : () -> index
    %6 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<8 × 16 × 4 × f32, #map0>
    %7 = "memref.subview"(%6, %0, %0, %0, %3, %4, %2, %1, %1, %1) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<8 × 16 × 4 × f32, #map0>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    %8 = "memref.load"(%7, %0, %0, %0) : (memref<? × ? × ? × f32, #map1>, index, index, index) -> f32
    %9 = "memref.subview"(%6, %0, %arg0, %0, %3, %4, %5, %1, %1, %1) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<8 × 16 × 4 × f32, #map0>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    "memref.store"(%8, %9, %0, %0, %0) : (f32, memref<? × ? × ? × f32, #map1>, index, index, index) -> ()
    %10 = "memref.alloc"(%arg0) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>} : (index) -> memref<? × 16 × 4 × f32, #map0>
    %11 = "memref.subview"(%10, %0, %0, %0, %3, %4, %5, %1, %1, %1) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<? × 16 × 4 × f32, #map0>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    "memref.store"(%8, %11, %0, %0, %0) : (f32, memref<? × ? × ? × f32, #map1>, index, index, index) -> ()
    %12 = "memref.subview"(%6, %1, %2, %3, %3, %4, %2, %1, %1, %1) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<8 × 16 × 4 × f32, #map0>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    "memref.store"(%8, %12, %0, %0, %0) : (f32, memref<? × ? × ? × f32, #map1>, index, index, index) -> ()
    %13 = "memref.subview"(%6, %0, %0, %0, %3, %4, %2, %2, %3, %4) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<8 × 16 × 4 × f32, #map0>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    "memref.store"(%8, %13, %0, %0, %0) : (f32, memref<? × ? × ? × f32, #map1>, index, index, index) -> ()
    %14 = "memref.subview"(%6, %arg0, %arg0, %arg0, %3, %4, %2, %arg1, %arg1, %arg1) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<8 × 16 × 4 × f32, #map0>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    "memref.store"(%8, %14, %arg1, %arg1, %arg1) : (f32, memref<? × ? × ? × f32, #map1>, index, index, index) -> ()
    %15 = "memref.subview"(%6, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %2, %3, %4) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<8 × 16 × 4 × f32, #map0>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    "memref.store"(%8, %15, %arg0, %arg0, %arg0) : (f32, memref<? × ? × ? × f32, #map1>, index, index, index) -> ()
    %16 = "memref.subview"(%6, %1, %2, %3, %arg1, %arg1, %arg1, %arg0, %arg0, %arg0) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<8 × 16 × 4 × f32, #map0>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    "memref.store"(%8, %16, %arg1, %arg1, %arg1) : (f32, memref<? × ? × ? × f32, #map1>, index, index, index) -> ()
    %17 = "memref.alloc"(%arg0, %arg0, %arg1) {operand_segment_sizes = dense<[3, 0]> : vector<2 × i32>} : (index, index, index) -> memref<? × ? × ? × f32>
    %18 = "memref.subview"(%17, %arg0, %arg0, %arg0, %3, %4, %2, %arg1, %arg1, %arg1) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<? × ? × ? × f32>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    "memref.store"(%8, %18, %arg1, %arg1, %arg1) : (f32, memref<? × ? × ? × f32, #map1>, index, index, index) -> ()
    %19 = "memref.subview"(%17, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %2, %2, %2) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<? × ? × ? × f32>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    "memref.store"(%8, %19, %arg0, %arg0, %arg0) : (f32, memref<? × ? × ? × f32, #map1>, index, index, index) -> ()
    %20 = "memref.subview"(%17, %1, %1, %1, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (memref<? × ? × ? × f32>, index, index, index, index, index, index, index, index, index) -> memref<? × ? × ? × f32, #map1>
    "memref.store"(%8, %20, %arg0, %arg0, %arg0) : (f32, memref<? × ? × ? × f32, #map1>, index, index, index) -> ()
    %21 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<12 × 4 × f32>
    %22 = "arith.constant"() {value = 4 : index} : () -> index
    %23 = "memref.subview"(%21, %arg1, %arg1, %2, %22) {operand_segment_sizes = dense<[1, 2, 2, 0]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808], static_sizes = [-1, -1], static_strides = [1, 1]} : (memref<12 × 4 × f32>, index, index, index, index) -> memref<? × ? × f32, #map2>
    "memref.store"(%8, %23, %arg1, %arg1) : (f32, memref<? × ? × f32, #map2>, index, index) -> ()
    %24 = "memref.subview"(%21, %2, %22) {operand_segment_sizes = dense<[1, 2, 0, 0]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808], static_sizes = [12, 4], static_strides = [1, 1]} : (memref<12 × 4 × f32>, index, index) -> memref<12 × 4 × f32, #map2>
    "memref.store"(%8, %24, %arg1, %arg1) : (f32, memref<12 × 4 × f32, #map2>, index, index) -> ()
    %25 = "memref.dim"(%11, %0) : (memref<? × ? × ? × f32, #map1>, index) -> index
    %26 = "memref.dim"(%11, %1) : (memref<? × ? × ? × f32, #map1>, index) -> index
    "func.return"(%25, %26) : (index, index) -> ()
  }) {function_type = (index, index) -> (index, index), sym_name = "subview"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i16):
    %0 = "arith.index_cast"(%arg0) : (i16) -> index
    %1 = "arith.index_cast"(%0) : (index) -> i16
    "func.return"(%1) : (i16) -> ()
  }) {function_type = (i16) -> i16, sym_name = "index_cast"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 4 : index} : () -> index
    %1 = "arith.index_cast"(%0) : (index) -> i16
    %2 = "arith.constant"() {value = 4 : i16} : () -> i16
    %3 = "arith.index_cast"(%2) : (i16) -> index
    "func.return"(%1, %3) : (i16, index) -> ()
  }) {function_type = () -> (i16, index), sym_name = "index_cast_fold"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<100 × i32>):
    "affine.for"() ({
    ^bb0(%arg1: index):
      %0 = "affine.load"(%arg0, %arg1) {map = #map3} : (memref<100 × i32>, index) -> i32
      "affine.if"(%arg1) ({
        "affine.for"() ({
        ^bb0(%arg2: index):
          %2 = "affine.load"(%arg0, %arg2) {map = #map3} : (memref<100 × i32>, index) -> i32
          "prevent.dce"(%2) : (i32) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
        "affine.yield"() : () -> ()
      }, {
        "affine.yield"() : () -> ()
      }) {condition = #set} : (index) -> ()
      %1 = "affine.load"(%arg0, %arg1) {map = #map3} : (memref<100 × i32>, index) -> i32
      "affine.yield"() : () -> ()
    }) {lower_bound = #map4, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<100 × i32>) -> (), sym_name = "remove_dead_else"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.divsi"(%arg0, %0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "divi_signed_by_one"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.divui"(%arg0, %0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "divi_unsigned_by_one"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.constant"() {value = dense<1> : tensor<4 × 5 × i32>} : () -> tensor<4 × 5 × i32>
    %1 = "arith.divsi"(%arg0, %0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%1) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "tensor_divi_signed_by_one"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.constant"() {value = dense<1> : tensor<4 × 5 × i32>} : () -> tensor<4 × 5 × i32>
    %1 = "arith.divui"(%arg0, %0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%1) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "tensor_divi_unsigned_by_one"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.floordivsi"(%arg0, %0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "arith.floordivsi_by_one"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.constant"() {value = dense<1> : tensor<4 × 5 × i32>} : () -> tensor<4 × 5 × i32>
    %1 = "arith.floordivsi"(%arg0, %0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%1) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "tensor_arith.floordivsi_by_one"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.ceildivsi"(%arg0, %0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "arith.ceildivsi_by_one"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.constant"() {value = dense<1> : tensor<4 × 5 × i32>} : () -> tensor<4 × 5 × i32>
    %1 = "arith.ceildivsi"(%arg0, %0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%1) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "tensor_arith.ceildivsi_by_one"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.ceildivui"(%arg0, %0) : (i32, i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "arith.ceildivui_by_one"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<4 × 5 × i32>):
    %0 = "arith.constant"() {value = dense<1> : tensor<4 × 5 × i32>} : () -> tensor<4 × 5 × i32>
    %1 = "arith.ceildivui"(%arg0, %0) : (tensor<4 × 5 × i32>, tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>
    "func.return"(%1) : (tensor<4 × 5 × i32>) -> ()
  }) {function_type = (tensor<4 × 5 × i32>) -> tensor<4 × 5 × i32>, sym_name = "tensor_arith.ceildivui_by_one"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<4 × 5 × f32>, %arg1: index):
    %0 = "memref.cast"(%arg0) : (memref<4 × 5 × f32>) -> memref<? × ? × f32>
    %1 = "memref.subview"(%0, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1) {operand_segment_sizes = dense<[1, 2, 2, 2]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808], static_sizes = [-1, -1], static_strides = [-9223372036854775808, -9223372036854775808]} : (memref<? × ? × f32>, index, index, index, index, index, index) -> memref<? × ? × f32, #map>
    "func.return"(%1) : (memref<? × ? × f32, #map>) -> ()
  }) {function_type = (memref<4 × 5 × f32>, index) -> memref<? × ? × f32, #map>, sym_name = "memref_cast_folding_subview"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<16 × 16 × f32>, %arg1: index, %arg2: index):
    %0 = "memref.cast"(%arg0) : (memref<16 × 16 × f32>) -> memref<? × ? × f32>
    %1 = "memref.subview"(%0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [0, 0], static_sizes = [3, 4], static_strides = [1, 1]} : (memref<? × ? × f32>) -> memref<3 × 4 × f32, #map>
    "func.return"(%1) : (memref<3 × 4 × f32, #map>) -> ()
  }) {function_type = (memref<16 × 16 × f32>, index, index) -> memref<3 × 4 × f32, #map>, sym_name = "memref_cast_folding_subview_static"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<8 × 16 × 4 × f32>, %arg1: index, %arg2: index):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 2 : index} : () -> index
    %3 = "arith.constant"() {value = 7 : index} : () -> index
    %4 = "arith.constant"() {value = 11 : index} : () -> index
    %5 = "tensor.extract_slice"(%arg0, %0, %0, %0, %3, %4, %2, %1, %1, %1) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (tensor<8 × 16 × 4 × f32>, index, index, index, index, index, index, index, index, index) -> tensor<? × ? × ? × f32>
    %6 = "tensor.extract_slice"(%5, %0, %0, %0, %2, %arg1, %2, %1, %1, %1) {operand_segment_sizes = dense<[1, 3, 3, 3]> : vector<4 × i32>, static_offsets = [-9223372036854775808, -9223372036854775808, -9223372036854775808], static_sizes = [-1, -1, -1], static_strides = [-9223372036854775808, -9223372036854775808, -9223372036854775808]} : (tensor<? × ? × ? × f32>, index, index, index, index, index, index, index, index, index) -> tensor<? × ? × ? × f32>
    "func.return"(%6) : (tensor<? × ? × ? × f32>) -> ()
  }) {function_type = (tensor<8 × 16 × 4 × f32>, index, index) -> tensor<? × ? × ? × f32>, sym_name = "slice"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1):
    %0 = "arith.extui"(%arg0) : (i1) -> i8
    %1 = "arith.trunci"(%0) : (i8) -> i1
    "func.return"(%1) : (i1) -> ()
  }) {function_type = (i1) -> i1, sym_name = "fold_trunci"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i1>):
    %0 = "arith.extui"(%arg0) : (vector<4 × i1>) -> vector<4 × i8>
    %1 = "arith.trunci"(%0) : (vector<4 × i8>) -> vector<4 × i1>
    "func.return"(%1) : (vector<4 × i1>) -> ()
  }) {function_type = (vector<4 × i1>) -> vector<4 × i1>, sym_name = "fold_trunci_vector"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1):
    %0 = "arith.extui"(%arg0) : (i1) -> i8
    %1 = "arith.trunci"(%0) : (i8) -> i2
    "func.return"(%1) : (i2) -> ()
  }) {function_type = (i1) -> i2, sym_name = "do_not_fold_trunci"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: vector<4 × i1>):
    %0 = "arith.extui"(%arg0) : (vector<4 × i1>) -> vector<4 × i8>
    %1 = "arith.trunci"(%0) : (vector<4 × i8>) -> vector<4 × i2>
    "func.return"(%1) : (vector<4 × i2>) -> ()
  }) {function_type = (vector<4 × i1>) -> vector<4 × i2>, sym_name = "do_not_fold_trunci_vector"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1):
    %0 = "arith.extsi"(%arg0) : (i1) -> i8
    %1 = "arith.trunci"(%0) : (i8) -> i1
    "func.return"(%1) : (i1) -> ()
  }) {function_type = (i1) -> i1, sym_name = "fold_trunci_sexti"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<5 × f32>
    %1 = "bufferization.clone"(%0) : (memref<5 × f32>) -> memref<5 × f32>
    "memref.dealloc"(%1) : (memref<5 × f32>) -> ()
    "func.return"(%0) : (memref<5 × f32>) -> ()
  }) {function_type = () -> memref<5 × f32>, sym_name = "simple_clone_elimination"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2 × f32>, %arg4: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "memref.dealloc"(%0) : (memref<2 × f32>) -> ()
    %1 = "bufferization.clone"(%arg3) : (memref<2 × f32>) -> memref<2 × f32>
    %2 = "scf.for"(%arg0, %arg1, %arg2, %1) ({
    ^bb0(%arg5: index, %arg6: memref<2 × f32>):
      %3 = "arith.cmpi"(%arg5, %arg1) {predicate = 0 : i64} : (index, index) -> i1
      "memref.dealloc"(%arg6) : (memref<2 × f32>) -> ()
      %4 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
      %5 = "bufferization.clone"(%4) : (memref<2 × f32>) -> memref<2 × f32>
      "memref.dealloc"(%4) : (memref<2 × f32>) -> ()
      %6 = "bufferization.clone"(%5) : (memref<2 × f32>) -> memref<2 × f32>
      "memref.dealloc"(%5) : (memref<2 × f32>) -> ()
      "scf.yield"(%6) : (memref<2 × f32>) -> ()
    }) : (index, index, index, memref<2 × f32>) -> memref<2 × f32>
    "memref.copy"(%2, %arg4) : (memref<2 × f32>, memref<2 × f32>) -> ()
    "memref.dealloc"(%2) : (memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "clone_loop_alloc"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    %0 = "arith.cmpi"(%arg0, %arg1) {predicate = 0 : i64} : (index, index) -> i1
    %1 = "arith.cmpi"(%arg0, %arg1) {predicate = 0 : i64} : (index, index) -> i1
    %2 = "memref.alloc"(%arg0, %arg0) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
    %3 = "scf.if"(%1) ({
      %4 = "scf.if"(%0) ({
        %6 = "bufferization.clone"(%2) : (memref<? × ? × f32>) -> memref<? × ? × f32>
        "scf.yield"(%6) : (memref<? × ? × f32>) -> ()
      }, {
        %6 = "memref.alloc"(%arg0, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
        %7 = "bufferization.clone"(%6) : (memref<? × ? × f32>) -> memref<? × ? × f32>
        "memref.dealloc"(%6) : (memref<? × ? × f32>) -> ()
        "scf.yield"(%7) : (memref<? × ? × f32>) -> ()
      }) : (i1) -> memref<? × ? × f32>
      %5 = "bufferization.clone"(%4) : (memref<? × ? × f32>) -> memref<? × ? × f32>
      "memref.dealloc"(%4) : (memref<? × ? × f32>) -> ()
      "scf.yield"(%5) : (memref<? × ? × f32>) -> ()
    }, {
      %4 = "memref.alloc"(%arg1, %arg1) {operand_segment_sizes = dense<[2, 0]> : vector<2 × i32>} : (index, index) -> memref<? × ? × f32>
      %5 = "bufferization.clone"(%4) : (memref<? × ? × f32>) -> memref<? × ? × f32>
      "memref.dealloc"(%4) : (memref<? × ? × f32>) -> ()
      "scf.yield"(%5) : (memref<? × ? × f32>) -> ()
    }) : (i1) -> memref<? × ? × f32>
    "memref.dealloc"(%2) : (memref<? × ? × f32>) -> ()
    "func.return"(%3) : (memref<? × ? × f32>) -> ()
  }) {function_type = (index, index, index) -> memref<? × ? × f32>, sym_name = "clone_nested_region"} : () -> ()
}) : () -> ()


