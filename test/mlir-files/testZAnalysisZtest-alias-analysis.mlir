"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2 × f32>, %arg1: memref<2 × f32>):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloca_1"} : () -> memref<8 × 64 × f32>
    %1 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloca_2"} : () -> memref<8 × 64 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc_1"} : () -> memref<8 × 64 × f32>
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc_2"} : () -> memref<8 × 64 × f32>
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "simple", test.ptr = "func"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2 × f32>, %arg1: i1):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloca_1"} : () -> memref<8 × 64 × f32>
    %1 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloca_2"} : () -> memref<8 × 64 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc_1"} : () -> memref<8 × 64 × f32>
    "cf.cond_br"(%arg1, %0, %0)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, memref<8 × 64 × f32>, memref<8 × 64 × f32>) -> ()
  ^bb1(%3: memref<8 × 64 × f32>):  
    "cf.br"(%3)[^bb2] : (memref<8 × 64 × f32>) -> ()
  ^bb2(%4: memref<8 × 64 × f32>):  
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × f32>, i1) -> (), sym_name = "control_flow", test.ptr = "func"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2 × f32>, %arg1: i1):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloca_1"} : () -> memref<8 × 64 × f32>
    %1 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloca_2"} : () -> memref<8 × 64 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc_1"} : () -> memref<8 × 64 × f32>
    "cf.cond_br"(%arg1, %0, %2)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, memref<8 × 64 × f32>, memref<8 × 64 × f32>) -> ()
  ^bb1(%3: memref<8 × 64 × f32>):  
    "cf.br"(%3)[^bb2] : (memref<8 × 64 × f32>) -> ()
  ^bb2(%4: memref<8 × 64 × f32>):  
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × f32>, i1) -> (), sym_name = "control_flow_merge", test.ptr = "func"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2 × f32>, %arg1: i1):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloca_1"} : () -> memref<8 × 64 × f32>
    %1 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloca_2"} : () -> memref<8 × 64 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc_1"} : () -> memref<8 × 64 × f32>
    %3 = "scf.if"(%arg1) ({
      "scf.yield"(%0) : (memref<8 × 64 × f32>) -> ()
    }, {
      "scf.yield"(%0) : (memref<8 × 64 × f32>) -> ()
    }) {test.ptr = "if_alloca"} : (i1) -> memref<8 × 64 × f32>
    %4 = "scf.if"(%arg1) ({
      "scf.yield"(%0) : (memref<8 × 64 × f32>) -> ()
    }, {
      "scf.yield"(%1) : (memref<8 × 64 × f32>) -> ()
    }) {test.ptr = "if_alloca_merge"} : (i1) -> memref<8 × 64 × f32>
    %5 = "scf.if"(%arg1) ({
      "scf.yield"(%2) : (memref<8 × 64 × f32>) -> ()
    }, {
      "scf.yield"(%2) : (memref<8 × 64 × f32>) -> ()
    }) {test.ptr = "if_alloc"} : (i1) -> memref<8 × 64 × f32>
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × f32>, i1) -> (), sym_name = "region_control_flow", test.ptr = "func"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2 × f32>, %arg1: index, %arg2: index, %arg3: index):
    %0 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloca_1"} : () -> memref<8 × 64 × f32>
    %1 = "memref.alloca"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloca_2"} : () -> memref<8 × 64 × f32>
    %2 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc_1"} : () -> memref<8 × 64 × f32>
    %3 = "scf.for"(%arg1, %arg2, %arg3, %0) ({
    ^bb0(%arg4: index, %arg5: memref<8 × 64 × f32>):
      "scf.yield"(%arg5) : (memref<8 × 64 × f32>) -> ()
    }) {test.ptr = "for_alloca"} : (index, index, index, memref<8 × 64 × f32>) -> memref<8 × 64 × f32>
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × f32>, index, index, index) -> (), sym_name = "region_loop_control_flow", test.ptr = "func"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2 × f32>, %arg1: index):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc_1"} : () -> memref<8 × 64 × f32>
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "memref.alloca"(%arg1) {operand_segment_sizes = dense<[1, 0]> : vector<2 × i32>, test.ptr = "alloca_1"} : (index) -> memref<? × i8>
    %3 = "memref.view"(%2, %1) {test.ptr = "view"} : (memref<? × i8>, index) -> memref<8 × 64 × f32>
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × f32>, index) -> (), sym_name = "view_like", test.ptr = "func"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2 × f32>):
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>, test.ptr = "alloc_1"} : () -> memref<8 × 64 × f32>
    %1 = "arith.constant"() {test.ptr = "constant_1", value = 0 : index} : () -> index
    %2 = "arith.constant"() {test.ptr = "constant_2", value = 0 : index} : () -> index
    %3 = "arith.constant"() {test.ptr = "constant_3", value = 1 : index} : () -> index
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × f32>) -> (), sym_name = "constants", test.ptr = "func"} : () -> ()
}) : () -> ()


