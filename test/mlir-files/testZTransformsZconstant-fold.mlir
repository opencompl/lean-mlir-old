"builtin.module"() ({
^bb0:
}) : () -> ()


#map0 = affine_map<() -> (0)>
#map1 = affine_map<() -> (8)>
#map2 = affine_map<() -> (128)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<f32>):
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.for"() ({
      ^bb0(%arg2: index):
        %0 = "arith.constant"() {value = 4.500000e+00 : f32} : () -> f32
        %1 = "arith.constant"() {value = 1.500000e+00 : f32} : () -> f32
        %2 = "arith.addf"(%0, %1) : (f32, f32) -> f32
        "memref.store"(%2, %arg0) : (f32, memref<f32>) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map0, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<f32>) -> (), sym_name = "affine_for"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 4.500000e+00 : f32} : () -> f32
    %1 = "arith.constant"() {value = 1.500000e+00 : f32} : () -> f32
    %2 = "arith.addf"(%0, %1) : (f32, f32) -> f32
    "func.return"(%2) : (f32) -> ()
  }) {function_type = () -> f32, sym_name = "simple_addf"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = dense<4.500000e+00> : tensor<4 × f32>} : () -> tensor<4 × f32>
    %1 = "arith.constant"() {value = dense<1.500000e+00> : tensor<4 × f32>} : () -> tensor<4 × f32>
    %2 = "arith.addf"(%0, %1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    "func.return"(%2) : (tensor<4 × f32>) -> ()
  }) {function_type = () -> tensor<4 × f32>, sym_name = "addf_splat_tensor"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = dense<[1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00]> : tensor<4 × f32>} : () -> tensor<4 × f32>
    %1 = "arith.constant"() {value = dense<[1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00]> : tensor<4 × f32>} : () -> tensor<4 × f32>
    %2 = "arith.addf"(%0, %1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    "func.return"(%2) : (tensor<4 × f32>) -> ()
  }) {function_type = () -> tensor<4 × f32>, sym_name = "addf_dense_tensor"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = dense<[1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00]> : tensor<4 × f32>} : () -> tensor<4 × f32>
    %1 = "arith.constant"() {value = dense<1.500000e+00> : tensor<4 × f32>} : () -> tensor<4 × f32>
    %2 = "arith.addf"(%0, %1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    "func.return"(%2) : (tensor<4 × f32>) -> ()
  }) {function_type = () -> tensor<4 × f32>, sym_name = "addf_dense_and_splat_tensors"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.constant"() {value = 5 : i32} : () -> i32
    %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "simple_addi"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i32):
    %0 = "arith.constant"() {value = true} : () -> i1
    %1 = "arith.constant"() {value = -1 : i32} : () -> i32
    %2 = "arith.constant"() {value = 31 : i32} : () -> i32
    %3 = "arith.andi"(%arg0, %0) : (i1, i1) -> i1
    %4 = "arith.andi"(%arg1, %1) : (i32, i32) -> i32
    %5 = "arith.andi"(%4, %2) : (i32, i32) -> i32
    "func.return"(%3, %5) : (i1, i32) -> ()
  }) {function_type = (i1, i32) -> (i1, i32), sym_name = "simple_and"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() {value = 31 : index} : () -> index
    %1 = "arith.constant"() {value = -1 : index} : () -> index
    %2 = "arith.andi"(%arg0, %0) : (index, index) -> index
    %3 = "arith.andi"(%2, %1) : (index, index) -> index
    "func.return"(%3) : (index) -> ()
  }) {function_type = (index) -> index, sym_name = "and_index"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<2 × i32>):
    %0 = "arith.constant"() {value = dense<-1> : tensor<2 × i32>} : () -> tensor<2 × i32>
    %1 = "arith.constant"() {value = dense<31> : tensor<2 × i32>} : () -> tensor<2 × i32>
    %2 = "arith.constant"() {value = dense<[31, -1]> : tensor<2 × i32>} : () -> tensor<2 × i32>
    %3 = "arith.andi"(%arg0, %0) : (tensor<2 × i32>, tensor<2 × i32>) -> tensor<2 × i32>
    %4 = "arith.andi"(%3, %1) : (tensor<2 × i32>, tensor<2 × i32>) -> tensor<2 × i32>
    %5 = "arith.andi"(%4, %2) : (tensor<2 × i32>, tensor<2 × i32>) -> tensor<2 × i32>
    "func.return"(%5) : (tensor<2 × i32>) -> ()
  }) {function_type = (tensor<2 × i32>) -> tensor<2 × i32>, sym_name = "tensor_and"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: vector<2 × i32>):
    %0 = "arith.constant"() {value = dense<-1> : vector<2 × i32>} : () -> vector<2 × i32>
    %1 = "arith.constant"() {value = dense<31> : vector<2 × i32>} : () -> vector<2 × i32>
    %2 = "arith.constant"() {value = dense<[31, -1]> : vector<2 × i32>} : () -> vector<2 × i32>
    %3 = "arith.andi"(%arg0, %0) : (vector<2 × i32>, vector<2 × i32>) -> vector<2 × i32>
    %4 = "arith.andi"(%3, %1) : (vector<2 × i32>, vector<2 × i32>) -> vector<2 × i32>
    %5 = "arith.andi"(%4, %2) : (vector<2 × i32>, vector<2 × i32>) -> vector<2 × i32>
    "func.return"(%5) : (vector<2 × i32>) -> ()
  }) {function_type = (vector<2 × i32>) -> vector<2 × i32>, sym_name = "vector_and"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = dense<1> : vector<8 × i32>} : () -> vector<8 × i32>
    %1 = "arith.constant"() {value = dense<5> : vector<8 × i32>} : () -> vector<8 × i32>
    %2 = "arith.addi"(%0, %1) : (vector<8 × i32>, vector<8 × i32>) -> vector<8 × i32>
    "func.return"(%2) : (vector<8 × i32>) -> ()
  }) {function_type = () -> vector<8 × i32>, sym_name = "addi_splat_vector"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 4.500000e+00 : f32} : () -> f32
    %1 = "arith.constant"() {value = 1.500000e+00 : f32} : () -> f32
    %2 = "arith.subf"(%0, %1) : (f32, f32) -> f32
    "func.return"(%2) : (f32) -> ()
  }) {function_type = () -> f32, sym_name = "simple_subf"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = dense<4.500000e+00> : vector<4 × f32>} : () -> vector<4 × f32>
    %1 = "arith.constant"() {value = dense<1.500000e+00> : vector<4 × f32>} : () -> vector<4 × f32>
    %2 = "arith.subf"(%0, %1) : (vector<4 × f32>, vector<4 × f32>) -> vector<4 × f32>
    "func.return"(%2) : (vector<4 × f32>) -> ()
  }) {function_type = () -> vector<4 × f32>, sym_name = "subf_splat_vector"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 4 : i32} : () -> i32
    %1 = "arith.constant"() {value = 1 : i32} : () -> i32
    %2 = "arith.constant"() {value = 0 : i32} : () -> i32
    %3 = "arith.subi"(%0, %1) : (i32, i32) -> i32
    %4 = "arith.subi"(%arg0, %2) : (i32, i32) -> i32
    "func.return"(%3, %4) : (i32, i32) -> ()
  }) {function_type = (i32) -> (i32, i32), sym_name = "simple_subi"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = dense<4> : tensor<4 × i32>} : () -> tensor<4 × i32>
    %1 = "arith.constant"() {value = dense<1> : tensor<4 × i32>} : () -> tensor<4 × i32>
    %2 = "arith.subi"(%0, %1) : (tensor<4 × i32>, tensor<4 × i32>) -> tensor<4 × i32>
    "func.return"(%2) : (tensor<4 × i32>) -> ()
  }) {function_type = () -> tensor<4 × i32>, sym_name = "subi_splat_tensor"} : () -> ()
}) : () -> ()


#map0 = affine_map<(d0, d1)[s0] -> (d0 floordiv 128 + s0 + d1 mod 128)>
#map1 = affine_map<(d0, d1)[s0] -> ((s0 ceildiv 128) * 128)>
#map2 = affine_map<(d0) -> (42)>
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() {value = 177 : index} : () -> index
    %1 = "arith.constant"() {value = 211 : index} : () -> index
    %2 = "arith.constant"() {value = 1075 : index} : () -> index
    %3 = "affine.apply"(%0, %1, %2) {map = #map0} : (index, index, index) -> index
    %4 = "affine.apply"(%0, %1, %2) {map = #map1} : (index, index, index) -> index
    %5 = "affine.apply"(%arg0) {map = #map2} : (index) -> index
    "func.return"(%3, %4, %5) : (index, index, index) -> ()
  }) {function_type = (index) -> (index, index, index), sym_name = "affine_apply"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 4.500000e+00 : f32} : () -> f32
    %1 = "arith.constant"() {value = 1.500000e+00 : f32} : () -> f32
    %2 = "arith.mulf"(%0, %1) : (f32, f32) -> f32
    "func.return"(%2) : (f32) -> ()
  }) {function_type = () -> f32, sym_name = "simple_mulf"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = dense<4.500000e+00> : tensor<4 × f32>} : () -> tensor<4 × f32>
    %1 = "arith.constant"() {value = dense<1.500000e+00> : tensor<4 × f32>} : () -> tensor<4 × f32>
    %2 = "arith.mulf"(%0, %1) : (tensor<4 × f32>, tensor<4 × f32>) -> tensor<4 × f32>
    "func.return"(%2) : (tensor<4 × f32>) -> ()
  }) {function_type = () -> tensor<4 × f32>, sym_name = "mulf_splat_tensor"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.constant"() {value = 6 : i32} : () -> i32
    %2 = "arith.constant"() {value = 2 : i32} : () -> i32
    %3 = "arith.divsi"(%1, %2) : (i32, i32) -> i32
    %4 = "arith.constant"() {value = -2 : i32} : () -> i32
    %5 = "arith.divsi"(%1, %4) : (i32, i32) -> i32
    %6 = "arith.divsi"(%1, %0) : (i32, i32) -> i32
    "func.return"(%3, %5, %6) : (i32, i32, i32) -> ()
  }) {function_type = () -> (i32, i32, i32), sym_name = "simple_divi_signed"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = dense<0> : tensor<4 × i32>} : () -> tensor<4 × i32>
    %1 = "arith.constant"() {value = dense<6> : tensor<4 × i32>} : () -> tensor<4 × i32>
    %2 = "arith.constant"() {value = dense<2> : tensor<4 × i32>} : () -> tensor<4 × i32>
    %3 = "arith.divsi"(%1, %2) : (tensor<4 × i32>, tensor<4 × i32>) -> tensor<4 × i32>
    %4 = "arith.constant"() {value = dense<-2> : tensor<4 × i32>} : () -> tensor<4 × i32>
    %5 = "arith.divsi"(%1, %4) : (tensor<4 × i32>, tensor<4 × i32>) -> tensor<4 × i32>
    %6 = "arith.divsi"(%1, %0) : (tensor<4 × i32>, tensor<4 × i32>) -> tensor<4 × i32>
    "func.return"(%3, %5, %6) : (tensor<4 × i32>, tensor<4 × i32>, tensor<4 × i32>) -> ()
  }) {function_type = () -> (tensor<4 × i32>, tensor<4 × i32>, tensor<4 × i32>), sym_name = "divi_signed_splat_tensor"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.constant"() {value = 6 : i32} : () -> i32
    %2 = "arith.constant"() {value = 2 : i32} : () -> i32
    %3 = "arith.divui"(%1, %2) : (i32, i32) -> i32
    %4 = "arith.constant"() {value = -2 : i32} : () -> i32
    %5 = "arith.divui"(%1, %4) : (i32, i32) -> i32
    %6 = "arith.divui"(%1, %0) : (i32, i32) -> i32
    "func.return"(%3, %5, %6) : (i32, i32, i32) -> ()
  }) {function_type = () -> (i32, i32, i32), sym_name = "simple_divi_unsigned"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = dense<0> : tensor<4 × i32>} : () -> tensor<4 × i32>
    %1 = "arith.constant"() {value = dense<6> : tensor<4 × i32>} : () -> tensor<4 × i32>
    %2 = "arith.constant"() {value = dense<2> : tensor<4 × i32>} : () -> tensor<4 × i32>
    %3 = "arith.divui"(%1, %2) : (tensor<4 × i32>, tensor<4 × i32>) -> tensor<4 × i32>
    %4 = "arith.constant"() {value = dense<-2> : tensor<4 × i32>} : () -> tensor<4 × i32>
    %5 = "arith.divui"(%1, %4) : (tensor<4 × i32>, tensor<4 × i32>) -> tensor<4 × i32>
    %6 = "arith.divui"(%1, %0) : (tensor<4 × i32>, tensor<4 × i32>) -> tensor<4 × i32>
    "func.return"(%3, %5, %6) : (tensor<4 × i32>, tensor<4 × i32>, tensor<4 × i32>) -> ()
  }) {function_type = () -> (tensor<4 × i32>, tensor<4 × i32>, tensor<4 × i32>), sym_name = "divi_unsigned_splat_tensor"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.constant"() {value = 7 : i32} : () -> i32
    %2 = "arith.constant"() {value = 2 : i32} : () -> i32
    %3 = "arith.floordivsi"(%1, %2) : (i32, i32) -> i32
    %4 = "arith.constant"() {value = -2 : i32} : () -> i32
    %5 = "arith.floordivsi"(%1, %4) : (i32, i32) -> i32
    %6 = "arith.constant"() {value = -9 : i32} : () -> i32
    %7 = "arith.floordivsi"(%6, %2) : (i32, i32) -> i32
    %8 = "arith.constant"() {value = -13 : i32} : () -> i32
    %9 = "arith.floordivsi"(%8, %4) : (i32, i32) -> i32
    %10 = "arith.floordivsi"(%1, %0) : (i32, i32) -> i32
    "func.return"(%3, %5, %7, %9, %10) : (i32, i32, i32, i32, i32) -> ()
  }) {function_type = () -> (i32, i32, i32, i32, i32), sym_name = "simple_arith.floordivsi"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.constant"() {value = 7 : i32} : () -> i32
    %2 = "arith.constant"() {value = 2 : i32} : () -> i32
    %3 = "arith.ceildivsi"(%1, %2) : (i32, i32) -> i32
    %4 = "arith.constant"() {value = -2 : i32} : () -> i32
    %5 = "arith.ceildivsi"(%1, %4) : (i32, i32) -> i32
    %6 = "arith.constant"() {value = -9 : i32} : () -> i32
    %7 = "arith.ceildivsi"(%6, %2) : (i32, i32) -> i32
    %8 = "arith.constant"() {value = -15 : i32} : () -> i32
    %9 = "arith.ceildivsi"(%8, %4) : (i32, i32) -> i32
    %10 = "arith.ceildivsi"(%1, %0) : (i32, i32) -> i32
    "func.return"(%3, %5, %7, %9, %10) : (i32, i32, i32, i32, i32) -> ()
  }) {function_type = () -> (i32, i32, i32, i32, i32), sym_name = "simple_arith.ceildivsi"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.constant"() {value = 7 : i32} : () -> i32
    %2 = "arith.constant"() {value = 2 : i32} : () -> i32
    %3 = "arith.ceildivui"(%1, %2) : (i32, i32) -> i32
    %4 = "arith.constant"() {value = -2 : i32} : () -> i32
    %5 = "arith.ceildivui"(%1, %4) : (i32, i32) -> i32
    %6 = "arith.constant"() {value = -8 : i32} : () -> i32
    %7 = "arith.ceildivui"(%6, %2) : (i32, i32) -> i32
    %8 = "arith.constant"() {value = -15 : i32} : () -> i32
    %9 = "arith.ceildivui"(%8, %4) : (i32, i32) -> i32
    %10 = "arith.ceildivui"(%1, %0) : (i32, i32) -> i32
    "func.return"(%3, %5, %7, %9, %10) : (i32, i32, i32, i32, i32) -> ()
  }) {function_type = () -> (i32, i32, i32, i32, i32), sym_name = "simple_arith.ceildivui"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 5 : i32} : () -> i32
    %1 = "arith.constant"() {value = 2 : i32} : () -> i32
    %2 = "arith.constant"() {value = 1 : i32} : () -> i32
    %3 = "arith.constant"() {value = -2 : i32} : () -> i32
    %4 = "arith.remsi"(%0, %1) : (i32, i32) -> i32
    %5 = "arith.remsi"(%0, %3) : (i32, i32) -> i32
    %6 = "arith.remsi"(%arg0, %2) : (i32, i32) -> i32
    "func.return"(%4, %5, %6) : (i32, i32, i32) -> ()
  }) {function_type = (i32) -> (i32, i32, i32), sym_name = "simple_arith.remsi"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 5 : i32} : () -> i32
    %1 = "arith.constant"() {value = 2 : i32} : () -> i32
    %2 = "arith.constant"() {value = 1 : i32} : () -> i32
    %3 = "arith.constant"() {value = -2 : i32} : () -> i32
    %4 = "arith.remui"(%0, %1) : (i32, i32) -> i32
    %5 = "arith.remui"(%0, %3) : (i32, i32) -> i32
    %6 = "arith.remui"(%arg0, %2) : (i32, i32) -> i32
    "func.return"(%4, %5, %6) : (i32, i32, i32) -> ()
  }) {function_type = (i32) -> (i32, i32, i32), sym_name = "simple_arith.remui"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 4 : i32} : () -> i32
    %1 = "arith.constant"() {value = 2 : i32} : () -> i32
    %2 = "arith.muli"(%0, %1) : (i32, i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "muli"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = dense<4> : vector<4 × i32>} : () -> vector<4 × i32>
    %1 = "arith.constant"() {value = dense<2> : vector<4 × i32>} : () -> vector<4 × i32>
    %2 = "arith.muli"(%0, %1) : (vector<4 × i32>, vector<4 × i32>) -> vector<4 × i32>
    "func.return"(%2) : (vector<4 × i32>) -> ()
  }) {function_type = () -> vector<4 × i32>, sym_name = "muli_splat_vector"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<8 × 4 × f32>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "tensor.dim"(%arg0, %0) : (tensor<8 × 4 × f32>, index) -> index
    "func.return"(%1) : (index) -> ()
  }) {function_type = (tensor<8 × 4 × f32>) -> index, sym_name = "dim"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 42 : i32} : () -> i32
    %1 = "arith.constant"() {value = -1 : i32} : () -> i32
    %2 = "arith.cmpi"(%0, %1) {predicate = 0 : i64} : (i32, i32) -> i1
    %3 = "arith.cmpi"(%0, %1) {predicate = 1 : i64} : (i32, i32) -> i1
    %4 = "arith.cmpi"(%0, %1) {predicate = 2 : i64} : (i32, i32) -> i1
    %5 = "arith.cmpi"(%0, %1) {predicate = 3 : i64} : (i32, i32) -> i1
    %6 = "arith.cmpi"(%0, %1) {predicate = 4 : i64} : (i32, i32) -> i1
    %7 = "arith.cmpi"(%0, %1) {predicate = 5 : i64} : (i32, i32) -> i1
    %8 = "arith.cmpi"(%0, %1) {predicate = 6 : i64} : (i32, i32) -> i1
    %9 = "arith.cmpi"(%0, %1) {predicate = 7 : i64} : (i32, i32) -> i1
    %10 = "arith.cmpi"(%0, %1) {predicate = 8 : i64} : (i32, i32) -> i1
    %11 = "arith.cmpi"(%0, %1) {predicate = 9 : i64} : (i32, i32) -> i1
    "func.return"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11) : (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) -> ()
  }) {function_type = () -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1), sym_name = "cmpi"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 4.200000e+01 : f32} : () -> f32
    %1 = "arith.constant"() {value = -1.000000e+00 : f32} : () -> f32
    %2 = "arith.cmpf"(%0, %1) {predicate = 0 : i64} : (f32, f32) -> i1
    %3 = "arith.cmpf"(%0, %1) {predicate = 1 : i64} : (f32, f32) -> i1
    %4 = "arith.cmpf"(%0, %1) {predicate = 2 : i64} : (f32, f32) -> i1
    %5 = "arith.cmpf"(%0, %1) {predicate = 3 : i64} : (f32, f32) -> i1
    %6 = "arith.cmpf"(%0, %1) {predicate = 4 : i64} : (f32, f32) -> i1
    %7 = "arith.cmpf"(%0, %1) {predicate = 5 : i64} : (f32, f32) -> i1
    %8 = "arith.cmpf"(%0, %1) {predicate = 6 : i64} : (f32, f32) -> i1
    %9 = "arith.cmpf"(%0, %1) {predicate = 7 : i64} : (f32, f32) -> i1
    %10 = "arith.cmpf"(%0, %1) {predicate = 8 : i64} : (f32, f32) -> i1
    %11 = "arith.cmpf"(%0, %1) {predicate = 9 : i64} : (f32, f32) -> i1
    %12 = "arith.cmpf"(%0, %1) {predicate = 10 : i64} : (f32, f32) -> i1
    %13 = "arith.cmpf"(%0, %1) {predicate = 11 : i64} : (f32, f32) -> i1
    %14 = "arith.cmpf"(%0, %1) {predicate = 12 : i64} : (f32, f32) -> i1
    %15 = "arith.cmpf"(%0, %1) {predicate = 13 : i64} : (f32, f32) -> i1
    %16 = "arith.cmpf"(%0, %1) {predicate = 14 : i64} : (f32, f32) -> i1
    %17 = "arith.cmpf"(%0, %1) {predicate = 15 : i64} : (f32, f32) -> i1
    "func.return"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17) : (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) -> ()
  }) {function_type = () -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1), sym_name = "cmpf_normal_numbers"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 4.200000e+01 : f32} : () -> f32
    %1 = "arith.constant"() {value = 0xFFFFFFFF : f32} : () -> f32
    %2 = "arith.cmpf"(%0, %1) {predicate = 0 : i64} : (f32, f32) -> i1
    %3 = "arith.cmpf"(%0, %1) {predicate = 1 : i64} : (f32, f32) -> i1
    %4 = "arith.cmpf"(%0, %1) {predicate = 2 : i64} : (f32, f32) -> i1
    %5 = "arith.cmpf"(%0, %1) {predicate = 3 : i64} : (f32, f32) -> i1
    %6 = "arith.cmpf"(%0, %1) {predicate = 4 : i64} : (f32, f32) -> i1
    %7 = "arith.cmpf"(%0, %1) {predicate = 5 : i64} : (f32, f32) -> i1
    %8 = "arith.cmpf"(%0, %1) {predicate = 6 : i64} : (f32, f32) -> i1
    %9 = "arith.cmpf"(%0, %1) {predicate = 7 : i64} : (f32, f32) -> i1
    %10 = "arith.cmpf"(%0, %1) {predicate = 8 : i64} : (f32, f32) -> i1
    %11 = "arith.cmpf"(%0, %1) {predicate = 9 : i64} : (f32, f32) -> i1
    %12 = "arith.cmpf"(%0, %1) {predicate = 10 : i64} : (f32, f32) -> i1
    %13 = "arith.cmpf"(%0, %1) {predicate = 11 : i64} : (f32, f32) -> i1
    %14 = "arith.cmpf"(%0, %1) {predicate = 12 : i64} : (f32, f32) -> i1
    %15 = "arith.cmpf"(%0, %1) {predicate = 13 : i64} : (f32, f32) -> i1
    %16 = "arith.cmpf"(%0, %1) {predicate = 14 : i64} : (f32, f32) -> i1
    %17 = "arith.cmpf"(%0, %1) {predicate = 15 : i64} : (f32, f32) -> i1
    "func.return"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17) : (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) -> ()
  }) {function_type = () -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1), sym_name = "cmpf_nan"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 4.200000e+01 : f32} : () -> f32
    %1 = "arith.constant"() {value = 0x7F800000 : f32} : () -> f32
    %2 = "arith.cmpf"(%0, %1) {predicate = 0 : i64} : (f32, f32) -> i1
    %3 = "arith.cmpf"(%0, %1) {predicate = 1 : i64} : (f32, f32) -> i1
    %4 = "arith.cmpf"(%0, %1) {predicate = 2 : i64} : (f32, f32) -> i1
    %5 = "arith.cmpf"(%0, %1) {predicate = 3 : i64} : (f32, f32) -> i1
    %6 = "arith.cmpf"(%0, %1) {predicate = 4 : i64} : (f32, f32) -> i1
    %7 = "arith.cmpf"(%0, %1) {predicate = 5 : i64} : (f32, f32) -> i1
    %8 = "arith.cmpf"(%0, %1) {predicate = 6 : i64} : (f32, f32) -> i1
    %9 = "arith.cmpf"(%0, %1) {predicate = 7 : i64} : (f32, f32) -> i1
    %10 = "arith.cmpf"(%0, %1) {predicate = 8 : i64} : (f32, f32) -> i1
    %11 = "arith.cmpf"(%0, %1) {predicate = 9 : i64} : (f32, f32) -> i1
    %12 = "arith.cmpf"(%0, %1) {predicate = 10 : i64} : (f32, f32) -> i1
    %13 = "arith.cmpf"(%0, %1) {predicate = 11 : i64} : (f32, f32) -> i1
    %14 = "arith.cmpf"(%0, %1) {predicate = 12 : i64} : (f32, f32) -> i1
    %15 = "arith.cmpf"(%0, %1) {predicate = 13 : i64} : (f32, f32) -> i1
    %16 = "arith.cmpf"(%0, %1) {predicate = 14 : i64} : (f32, f32) -> i1
    %17 = "arith.cmpf"(%0, %1) {predicate = 15 : i64} : (f32, f32) -> i1
    "func.return"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17) : (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) -> ()
  }) {function_type = () -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1), sym_name = "cmpf_inf"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "func.func"() ({
      %0 = "arith.constant"() {value = 1 : i32} : () -> i32
      %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
      "foo.yield"(%1) : (i32) -> ()
    }) {function_type = () -> (), sym_name = "isolated_op"} : () -> ()
    "foo.unknown_region"() ({
      %0 = "arith.constant"() {value = 1 : i32} : () -> i32
      %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
      "foo.yield"(%1) : (i32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "nested_isolated_region"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.one_region_op"() ({
      %0 = "arith.constant"() {value = 1 : i32} : () -> i32
      %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
      "foo.yield"(%1) : (i32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "custom_insertion_position"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<f32>):
    %0 = "memref.subview"(%arg0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [], static_sizes = [], static_strides = []} : (memref<f32>) -> memref<f32>
    "func.return"(%0) : (memref<f32>) -> ()
  }) {function_type = (memref<f32>) -> memref<f32>, sym_name = "subview_scalar_fold"} : () -> ()
}) : () -> ()


