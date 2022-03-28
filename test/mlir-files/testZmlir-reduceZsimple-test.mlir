"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: memref<2 × f32>, %arg2: memref<2 × f32>):
    "cf.cond_br"(%arg0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  
    "cf.br"(%arg1)[^bb3] : (memref<2 × f32>) -> ()
  ^bb2:  
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "cf.br"(%0)[^bb3] : (memref<2 × f32>) -> ()
  ^bb3(%1: memref<2 × f32>):  
    "func.return"() : () -> ()
  }) {function_type = (i1, memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "simple1"} : () -> ()
}) : () -> ()


