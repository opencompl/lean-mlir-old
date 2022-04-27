

























"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (i32, i64) -> f32, sym_name = "foo", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "bar", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (i1, index, f32), sym_name = "baz", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "missingReturn", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (i0, i1, i2, i4, i7, i87) -> (i1, index, i19), sym_name = "int_types", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (si2, si4) -> (si7, si1023), sym_name = "sint_types", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (ui2, ui4) -> (ui7, ui1023), sym_name = "uint_types", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (f80, f128) -> (), sym_name = "float_types", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (vector<f32>, vector<1 × f32>, vector<2 × 4 × f32>) -> (), sym_name = "vectors", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (tensor<* × f32>, tensor<*xvector<2 × 4 × f32>>, tensor<1 × ? × 4 × ? × ? × i32>, tensor<i8>) -> (), sym_name = "tensors", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (tensor<16 × 32 × f64, "sparse">) -> (), sym_name = "tensor_encoding", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (tensor<9223372036854775807 × f32>) -> (), sym_name = "large_shape_dimension", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = ((memref<1 × ? × 4 × ? × ? × i32, #map0>, memref<8 × i8>) -> (), () -> ()) -> (), sym_name = "functions", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × 8 × i8, 1>) -> (), sym_name = "memrefs2", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × 8 × i8>) -> (), sym_name = "memrefs3", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × i8>) -> (), sym_name = "memrefs_drop_triv_id_inline", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × i8>) -> (), sym_name = "memrefs_drop_triv_id_inline0", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × i8, 1>) -> (), sym_name = "memrefs_drop_triv_id_inline1", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<5 × 6 × 7 × f32>) -> (), sym_name = "memrefs_nomap_nospace", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<5 × 6 × 7 × f32, #map1>) -> (), sym_name = "memrefs_map_nospace", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<5 × 6 × 7 × f32, 3>) -> (), sym_name = "memrefs_nomap_intspace", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<5 × 6 × 7 × f32, #map1, 5>) -> (), sym_name = "memrefs_map_intspace", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<5 × 6 × 7 × f32, "local">) -> (), sym_name = "memrefs_nomap_strspace", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<5 × 6 × 7 × f32, #map1, "private">) -> (), sym_name = "memrefs_map_strspace", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<5 × 6 × 7 × f32, {memSpace = "special", subInde ×  = 1 : i64}>) -> (), sym_name = "memrefs_nomap_dictspace", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<5 × 6 × 7 × f32, #map1, {memSpace = "special", subInde ×  = 3 : i64}>) -> (), sym_name = "memrefs_map_dictspace", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (complex<i1>) -> complex<f32>, sym_name = "complex_types", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × ? × inde × >) -> (), sym_name = "memref_with_index_elems", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × ? × comple × <f32>>) -> (), sym_name = "memref_with_complex_elems", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × ? × vector<10 × f32>>) -> (), sym_name = "memref_with_vector_elems", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × ? × !test.memref_element>) -> (), sym_name = "memref_with_custom_elem", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1xmemref<1 × f64>>) -> (), sym_name = "memref_of_memref", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1xmemref<* × f32>>) -> (), sym_name = "memref_of_unranked_memref", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<*xmemref<1 × f32>>) -> (), sym_name = "unranked_memref_of_memref", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<*xmemref<* × i32>>) -> (), sym_name = "unranked_memref_of_unranked_memref", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<* × comple × <f32>>) -> (), sym_name = "unranked_memref_with_complex_elems", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<* × inde × >) -> (), sym_name = "unranked_memref_with_index_elems", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<*xvector<10 × f32>>) -> (), sym_name = "unranked_memref_with_vector_elems", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    %1 = "foo"() : () -> i64
    %2:3 = "bar"(%1) : (i64) -> (i1, i1, i1)
    "func.return"(%2#1) : (i1) -> ()
  }) {function_type = (i32, f32) -> i1, sym_name = "simpleCFG"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i64):
    %1:3 = "bar"(%arg1) : (i64) -> (i1, i1, i1)
    "func.return"() : () -> ()
  }) {function_type = (i32, i64) -> (), sym_name = "simpleCFGUsingBBArgs"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "block_label_empty_list"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  ^bb1:  
    "cf.br"()[^bb3] : () -> ()
  ^bb2:  
    "cf.br"()[^bb2] : () -> ()
  ^bb3:  
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "multiblock"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "emptyMLF"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    %1 = "foo"(%arg0) : (i1) -> i2
    "func.return"(%1) : (i2) -> ()
  }) {function_type = (i1) -> i2, sym_name = "func_with_one_arg"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: f16, %arg1: i8):
    %1:2 = "foo"(%arg0, %arg1) : (f16, i8) -> (i1, i32)
    "func.return"(%1#0, %1#1) : (i1, i32) -> ()
  }) {function_type = (f16, i8) -> (i1, i32), sym_name = "func_with_two_args"} : () -> ()
  "func.func"() ({
    %1 = "func.constant"() {value = @emptyMLF} : () -> (() -> ())
    "func.return"(%1) : (() -> ()) -> ()
  }) {function_type = () -> (() -> ()), sym_name = "second_order_func"} : () -> ()
  "func.func"() ({
    %1 = "func.constant"() {value = @second_order_func} : () -> (() -> (() -> ()))
    "func.return"(%1) : (() -> (() -> ())) -> ()
  }) {function_type = () -> (() -> (() -> ())), sym_name = "third_order_func"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: () -> ()):
    "func.return"(%arg0) : (() -> ()) -> ()
  }) {function_type = (() -> ()) -> (() -> ()), sym_name = "identity_functor"} : () -> ()
  "func.func"() ({
    %1 = "foo"() : () -> i64
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "doo"() : () -> f32
      "bar"(%1, %2) : (i64, f32) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "func_ops_in_loop"} : () -> ()
  "func.func"() ({
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map4} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 2 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "loops"} : () -> ()
  "func.func"() ({
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "foo"(%arg0, %arg1) : (index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : () -> ()
      "boo"() : () -> ()
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.for"() ({
        ^bb0(%arg2: index):
          "goo"() : () -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "complex_loops"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: memref<? × ? × i32>):
    %1 = "arith.constant"() {value = 0 : i32} : () -> i32
    "affine.for"(%arg0) ({
    ^bb0(%arg2: index):
      "affine.for"(%arg2, %arg0) ({
      ^bb0(%arg3: index):
        "memref.store"(%1, %arg1, %arg2, %arg3) : (i32, memref<? × ? × i32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map6, step = 1 : index, upper_bound = #map7} : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map7} : (index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, memref<? × ? × i32>) -> (), sym_name = "triang_loop"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: memref<100 × f32>):
    "affine.for"(%arg0, %arg1) ({
    ^bb0(%arg3: index):
      "foo"(%arg2, %arg3) : (memref<100 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map8, step = 1 : index, upper_bound = #map9} : (index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, memref<100 × f32>) -> (), sym_name = "minmax_loop"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %1 = "foo"(%arg0) : (index) -> index
    "affine.for"(%1, %arg0) ({
    ^bb0(%arg1: index):
      "affine.for"(%arg1) ({
      ^bb0(%arg2: index):
        %2 = "affine.apply"(%arg1, %arg2, %1) {map = #map10} : (index, index, index) -> index
        %3 = "affine.apply"(%arg1, %arg2, %1) {map = #map11} : (index, index, index) -> index
        "affine.for"(%2, %arg1, %arg0, %3, %arg2, %1) ({
        ^bb0(%arg3: index):
          "foo"(%arg1, %arg2, %arg3) : (index, index, index) -> ()
          %4 = "arith.constant"() {value = 30 : index} : () -> index
          %5 = "affine.apply"(%arg0, %4) {map = #map12} : (index, index) -> index
          "affine.for"(%arg1, %5, %arg3, %4) ({
          ^bb0(%arg4: index):
            "bar"(%arg4) : (index) -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map13, step = 1 : index, upper_bound = #map13} : (index, index, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map14, step = 1 : index, upper_bound = #map14} : (index, index, index, index, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map6, step = 1 : index, upper_bound = #map15} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map7, step = 1 : index, upper_bound = #map7} : (index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "loop_bounds"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %1 = "arith.constant"() {value = 200 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.if"(%arg1, %arg0, %1) ({
        %2 = "arith.constant"() {value = 1 : i32} : () -> i32
        %3 = "add"(%2, %arg1) : (i32, index) -> i32
        %4 = "mul"(%3, %3) : (i32, i32) -> i32
        "affine.yield"() : () -> ()
      }, {
        "affine.if"(%arg1, %arg0) ({
          %2 = "arith.constant"() {value = 1 : index} : () -> index
          %3 = "affine.apply"(%arg1, %arg1, %2) {map = #map14} : (index, index, index) -> index
          "affine.yield"() : () -> ()
        }, {
          %2 = "arith.constant"() {value = 3 : i32} : () -> i32
          "affine.yield"() : () -> ()
        }) {condition = #set0} : (index, index) -> ()
        "affine.yield"() : () -> ()
      }) {condition = #set1} : (index, index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "ifinst"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %1 = "arith.constant"() {value = 200 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg1: index):
      "affine.if"(%arg1, %arg0, %1) ({
        %2 = "arith.constant"() {value = 1 : i32} : () -> i32
        %3 = "add"(%2, %arg1) : (i32, index) -> i32
        %4 = "mul"(%3, %3) : (i32, i32) -> i32
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set1} : (index, index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "simple_ifinst"} : () -> ()
  "func.func"() ({
    "foo"() : () -> ()
    "foo"() {a = 1 : i64, b = -423 : i64, c = [true, false], d = 1.60000001 : f64} : () -> ()
    "foo"() {map1 = #map6} : () -> ()
    "foo"() {map2 = #map16} : () -> ()
    "foo"() {map12 = [#map6, #map16]} : () -> ()
    "foo"() {set1 = #set2} : () -> ()
    "foo"() {set2 = #set3} : () -> ()
    "foo"() {set12 = [#set2, #set4]} : () -> ()
    "foo"() {dictionary = {bool = true, fn = @ifinst}} : () -> ()
    "foo"() {dictionary = {bar = false, bool = true, fn = @ifinst}} : () -> ()
    "foo"() {d = 1.000000e-09 : f64, func = [], i123 = 7 : i64, if = "foo"} : () -> ()
    "foo"() {fn = @attributes, if = @ifinst} : () -> ()
    "foo"() {int = 0 : i42} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "attributes"} : () -> ()
  "func.func"() ({
    %1:2 = "foo"() : () -> (i1, i17)
    "cf.br"()[^bb2] : () -> ()
  ^bb1:  
    %2:2 = "baz"(%3#1, %3#0, %1#1) : (f32, i11, i17) -> (i16, i8)
    "func.return"(%2#0, %2#1) : (i16, i8) -> ()
  ^bb2:  
    %3:2 = "bar"(%1#0, %1#1) : (i1, i17) -> (i11, f32)
    "cf.br"()[^bb1] : () -> ()
  }) {function_type = () -> (i16, i8), sym_name = "ssa_values"} : () -> ()
  "func.func"() ({
    %1:2 = "foo"() : () -> (i1, i17)
    "cf.br"(%1#1, %1#0)[^bb1] : (i17, i1) -> ()
  ^bb1(%2: i17, %3: i1):  
    %4:2 = "baz"(%2, %3, %1#1) : (i17, i1, i17) -> (i16, i8)
    "func.return"(%4#0, %4#1) : (i16, i8) -> ()
  }) {function_type = () -> (i16, i8), sym_name = "bbargs"} : () -> ()
  "func.func"() ({
    %1:2 = "foo"() : () -> (i1, i17)
    "cf.br"(%1#0, %1#1)[^bb1] : (i1, i17) -> ()
  ^bb1(%2: i1, %3: i17):  
    "cf.cond_br"(%2, %3, %2, %3)[^bb2, ^bb3] {operand_segment_sizes = dense<[1, 1, 2]> : vector<3 × i32>} : (i1, i17, i1, i17) -> ()
  ^bb2(%4: i17):  
    %5 = "arith.constant"() {value = true} : () -> i1
    "func.return"(%5, %4) : (i1, i17) -> ()
  ^bb3(%6: i1, %7: i17):  
    "func.return"(%6, %7) : (i1, i17) -> ()
  }) {function_type = () -> (i1, i17), sym_name = "verbose_terminators"} : () -> ()
  "func.func"() ({
    %1 = "foo"() : () -> i1
    %2 = "bar"() : () -> i32
    %3 = "bar"() : () -> i64
    "cf.cond_br"(%1, %2, %3)[^bb1, ^bb2] {operand_segment_sizes = dense<1> : vector<3 × i32>} : (i1, i32, i64) -> ()
  ^bb1(%4: i32):  
    "cf.br"(%3)[^bb2] : (i64) -> ()
  ^bb2(%5: i64):  
    %6 = "foo"() : () -> i32
    "func.return"(%6) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "condbr_simple"} : () -> ()
  "func.func"() ({
    %1 = "foo"() : () -> i1
    %2 = "bar"() : () -> i32
    %3 = "bar"() : () -> i64
    "cf.cond_br"(%1, %2, %3, %3, %2, %2)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 2, 3]> : vector<3 × i32>} : (i1, i32, i64, i64, i32, i32) -> ()
  ^bb1(%4: i32, %5: i64):  
    "func.return"(%4) : (i32) -> ()
  ^bb2(%6: i64, %7: i32, %8: i32):  
    %9 = "foo"() : () -> i32
    "func.return"(%9) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "condbr_moarargs"} : () -> ()
  "func.func"() ({
    %1 = "arith.constant"() {value = 42 : i32} : () -> i32
    %2 = "arith.constant"() {value = 17 : i23} : () -> i23
    %3 = "arith.constant"() {value = 17 : i23} : () -> i23
    %4 = "arith.constant"() {value = true} : () -> i1
    %5 = "arith.constant"() {value = false} : () -> i1
    %6 = "arith.constant"() {value = 3890 : i32} : () -> i32
    "func.return"(%1, %2, %3, %4, %5) : (i32, i23, i23, i1, i1) -> ()
  }) {function_type = () -> (i32, i23, i23, i1, i1), sym_name = "constants"} : () -> ()
  "func.func"() ({
    "foo"() {bar = tensor<* × f32>} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "typeattr"} : () -> ()
  "func.func"() ({
    "foo"() {bar = "a\22quoted\22string"} : () -> ()
    "foo"() {bar = "typed_string" : !foo.string} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "stringquote"} : () -> ()
  "func.func"() ({
    "foo"() {unitAttr} : () -> ()
    "foo"() {unitAttr} : () -> ()
    "foo"() {nested = {unitAttr}} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "unitAttrs"} : () -> ()
  "func.func"() ({
    "foo"() {a = 4.00000000 : f64, b = 2.00000000 : f64, c = 7.10000000 : f64, d = -0.00000000 : f64} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "floatAttrs"} : () -> ()
  "func.func"() ({
  }) {dialect.a = "a\22quoted\22string", dialect.b = 4.00000000 : f64, dialect.c = tensor<* × f32>, function_type = () -> (), sym_name = "externalfuncattr", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "funcattrempty", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {dialect.a = "a\22quoted\22string", dialect.b = 4.00000000 : f64, dialect.c = tensor<* × f32>, function_type = () -> (), sym_name = "funcattr", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "funcattrwithblock"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    "affine.for"() ({
    ^bb0(%arg2: index):
      "affine.for"(%arg1) ({
      ^bb0(%arg3: index):
        "affine.for"(%arg2) ({
        ^bb0(%arg4: index):
          "affine.for"(%arg2, %arg1) ({
          ^bb0(%arg5: index):
            "affine.for"(%arg1, %arg0) ({
            ^bb0(%arg6: index):
              "affine.for"(%arg0) ({
              ^bb0(%arg7: index):
                %1 = "arith.constant"() {value = 42 : i32} : () -> i32
                "affine.yield"() : () -> ()
              }) {lower_bound = #map15, step = 1 : index, upper_bound = #map17} : (index) -> ()
              "affine.yield"() : () -> ()
            }) {lower_bound = #map15, step = 1 : index, upper_bound = #map18} : (index, index) -> ()
            "affine.yield"() : () -> ()
          }) {lower_bound = #map15, step = 1 : index, upper_bound = #map19} : (index, index) -> ()
          "affine.yield"() : () -> ()
        }) {lower_bound = #map15, step = 1 : index, upper_bound = #map6} : (index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map15, step = 1 : index, upper_bound = #map7} : (index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map15, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index) -> (), sym_name = "funcsimplemap"} : () -> ()
  "func.func"() ({
    "splatBoolTensor"() {bar = dense<false> : tensor<i1>} : () -> ()
    "splatUIntTensor"() {bar = dense<222> : tensor<2 × 1 × 4 × ui8>} : () -> ()
    "splatIntTensor"() {bar = dense<5> : tensor<2 × 1 × 4 × i32>} : () -> ()
    "splatFloatTensor"() {bar = dense<-5.00000000> : tensor<2 × 1 × 4 × f32>} : () -> ()
    "splatIntVector"() {bar = dense<5> : vector<2 × 1 × 4 × i64>} : () -> ()
    "splatFloatVector"() {bar = dense<-5.00000000> : vector<2 × 1 × 4 × f16>} : () -> ()
    "splatIntScalar"() {bar = dense<5> : tensor<i9>} : () -> ()
    "splatFloatScalar"() {bar = dense<-5.00000000> : tensor<f16>} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "splattensorattr"} : () -> ()
  "func.func"() ({
    "fooi3"() {bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2 × 1 × 4 × i3>} : () -> ()
    "fooi6"() {bar = dense<[[[5, -6, 1, 2]], [[7, 8, 3, 4]]]> : tensor<2 × 1 × 4 × i6>} : () -> ()
    "fooi8"() {bar = dense<5> : tensor<1 × 1 × 1 × i8>} : () -> ()
    "fooi13"() {bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2 × 1 × 4 × i13>} : () -> ()
    "fooi16"() {bar = dense<-5> : tensor<1 × 1 × 1 × i16>} : () -> ()
    "fooi23"() {bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2 × 1 × 4 × i23>} : () -> ()
    "fooi32"() {bar = dense<5> : tensor<1 × 1 × 1 × i32>} : () -> ()
    "fooi33"() {bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2 × 1 × 4 × i33>} : () -> ()
    "fooi43"() {bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2 × 1 × 4 × i43>} : () -> ()
    "fooi53"() {bar = dense<[[[1, -2, 1, 2]], [[0, 2, -1, 2]]]> : tensor<2 × 1 × 4 × i53>} : () -> ()
    "fooi64"() {bar = dense<[[[1, -2, 1, 2]], [[0, 3, -1, 2]]]> : tensor<2 × 1 × 4 × i64>} : () -> ()
    "fooi64"() {bar = dense<-5> : tensor<1 × 1 × 1 × i64>} : () -> ()
    "fooi67"() {bar = dense<[[[-5, 4, 6, 2]]]> : vector<1 × 1 × 4 × i67>} : () -> ()
    "foo2"() {bar = dense<> : tensor<0 × i32>} : () -> ()
    "foo2"() {bar = dense<> : tensor<1 × 0 × i32>} : () -> ()
    "foo2"() {bar = dense<> : tensor<0 × 512 × 512 × i32>} : () -> ()
    "foo3"() {bar = dense<[[[5, -6, 1, 2]], [[7, 8, 3, 4]]]> : tensor<2 × 1 × 4 × i32>} : () -> ()
    "float1"() {bar = dense<5.00000000> : tensor<1 × 1 × 1 × f32>} : () -> ()
    "float2"() {bar = dense<> : tensor<0 × f32>} : () -> ()
    "float2"() {bar = dense<> : tensor<1 × 0 × f32>} : () -> ()
    "bfloat16"() {bar = dense<[[[-5.00000000, 6.00000000, 1.00000000, 2.00000000]], [[7.00000000, -8.00000000, 3.00000000, 4.00000000]]]> : tensor<2 × 1 × 4 × bf16>} : () -> ()
    "float16"() {bar = dense<[[[-5.00000000, 6.00000000, 1.00000000, 2.00000000]], [[7.00000000, -8.00000000, 3.00000000, 4.00000000]]]> : tensor<2 × 1 × 4 × f16>} : () -> ()
    "float32"() {bar = dense<[[[-5.00000000, 6.00000000, 1.00000000, 2.00000000]], [[7.00000000, -8.00000000, 3.00000000, 4.00000000]]]> : tensor<2 × 1 × 4 × f32>} : () -> ()
    "float64"() {bar = dense<[[[-5.00000000, 6.00000000, 1.00000000, 2.00000000]], [[7.00000000, -8.00000000, 3.00000000, 4.00000000]]]> : tensor<2 × 1 × 4 × f64>} : () -> ()
    "intscalar"() {bar = dense<1> : tensor<i32>} : () -> ()
    "floatscalar"() {bar = dense<5.00000000> : tensor<f32>} : () -> ()
    "index"() {bar = dense<1> : tensor<inde × >} : () -> ()
    "index"() {bar = dense<[1, 2]> : tensor<2 × inde × >} : () -> ()
    "complex_attr"() {bar = dense<(1,1)> : tensor<comple × <i64>>} : () -> ()
    "complex_attr"() {bar = dense<[(1,1), (2,2)]> : tensor<2 × comple × <i64>>} : () -> ()
    "complex_attr"() {bar = dense<(1.00000000,0.00000000)> : tensor<comple × <f32>>} : () -> ()
    "complex_attr"() {bar = dense<[(1.00000000,0.00000000), (2.00000000,2.00000000)]> : tensor<2 × comple × <f32>>} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "densetensorattr"} : () -> ()
  "func.func"() ({
    "fooi8"() {bar = dense<5> : vector<1 × 1 × 1 × i8>} : () -> ()
    "fooi16"() {bar = dense<-5> : vector<1 × 1 × 1 × i16>} : () -> ()
    "foo32"() {bar = dense<5> : vector<1 × 1 × 1 × i32>} : () -> ()
    "fooi64"() {bar = dense<-5> : vector<1 × 1 × 1 × i64>} : () -> ()
    "foo3"() {bar = dense<[[[5, -6, 1, 2]], [[7, 8, 3, 4]]]> : vector<2 × 1 × 4 × i32>} : () -> ()
    "float1"() {bar = dense<5.00000000> : vector<1 × 1 × 1 × f32>} : () -> ()
    "bfloat16"() {bar = dense<[[[-5.00000000, 6.00000000, 1.00000000, 2.00000000]], [[7.00000000, -8.00000000, 3.00000000, 4.00000000]]]> : vector<2 × 1 × 4 × bf16>} : () -> ()
    "float16"() {bar = dense<[[[-5.00000000, 6.00000000, 1.00000000, 2.00000000]], [[7.00000000, -8.00000000, 3.00000000, 4.00000000]]]> : vector<2 × 1 × 4 × f16>} : () -> ()
    "float32"() {bar = dense<[[[-5.00000000, 6.00000000, 1.00000000, 2.00000000]], [[7.00000000, -8.00000000, 3.00000000, 4.00000000]]]> : vector<2 × 1 × 4 × f32>} : () -> ()
    "float64"() {bar = dense<[[[-5.00000000, 6.00000000, 1.00000000, 2.00000000]], [[7.00000000, -8.00000000, 3.00000000, 4.00000000]]]> : vector<2 × 1 × 4 × f64>} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "densevectorattr"} : () -> ()
  "func.func"() ({
    "fooi8"() {bar = sparse<0, -2> : tensor<1 × 1 × 1 × i8>} : () -> ()
    "fooi16"() {bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2, -1, 5]> : tensor<2 × 2 × 2 × i16>} : () -> ()
    "fooi32"() {bar = sparse<> : tensor<1 × 1 × i32>} : () -> ()
    "fooi64"() {bar = sparse<0, -1> : tensor<1 × i64>} : () -> ()
    "foo2"() {bar = sparse<> : tensor<0 × i32>} : () -> ()
    "foo3"() {bar = sparse<> : tensor<i32>} : () -> ()
    "foof16"() {bar = sparse<0, -2.00000000> : tensor<1 × 1 × 1 × f16>} : () -> ()
    "foobf16"() {bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2.00000000, -1.00000000, 5.00000000]> : tensor<2 × 2 × 2 × bf16>} : () -> ()
    "foof32"() {bar = sparse<> : tensor<1 × 0 × 1 × f32>} : () -> ()
    "foof64"() {bar = sparse<0, -1.00000000> : tensor<1 × f64>} : () -> ()
    "foof320"() {bar = sparse<> : tensor<0 × f32>} : () -> ()
    "foof321"() {bar = sparse<> : tensor<f32>} : () -> ()
    "foostr"() {bar = sparse<0, "foo"> : tensor<1 × 1 × 1 × !unknown<"">>} : () -> ()
    "foostr"() {bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], ["a", "b", "c"]> : tensor<2 × 2 × 2 × !unknown<"">>} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "sparsetensorattr"} : () -> ()
  "func.func"() ({
    "fooi8"() {bar = sparse<0, -2> : vector<1 × 1 × 1 × i8>} : () -> ()
    "fooi16"() {bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2, -1, 5]> : vector<2 × 2 × 2 × i16>} : () -> ()
    "fooi32"() {bar = sparse<> : vector<1 × 1 × i32>} : () -> ()
    "fooi64"() {bar = sparse<0, -1> : vector<1 × i64>} : () -> ()
    "foof16"() {bar = sparse<0, -2.00000000> : vector<1 × 1 × 1 × f16>} : () -> ()
    "foobf16"() {bar = sparse<[[1, 1, 0], [0, 1, 0], [0, 0, 1]], [2.00000000, -1.00000000, 5.00000000]> : vector<2 × 2 × 2 × bf16>} : () -> ()
    "foof64"() {bar = sparse<0, -1.00000000> : vector<1 × f64>} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "sparsevectorattr"} : () -> ()
  "func.func"() ({
    %1 = "foo"() : () -> !bar<"">
    %2 = "foo"() : () -> !bar.baz
    "func.return"(%1) : (!bar<"">) -> ()
  }) {function_type = () -> !bar<"">, sym_name = "unknown_dialect_type"} : () -> ()
  "func.func"() ({
    %1 = "foo"() : () -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "type_alias"} : () -> ()
  "func.func"() ({
    "affine.if"() ({
      "affine.yield"() : () -> ()
    }, {
    }) {condition = #set5} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "no_integer_set_constraints"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index):
    %1 = "arith.constant"() {value = 200 : index} : () -> index
    "affine.if"(%1, %arg0, %1) ({
      %2 = "add"(%1, %arg0) : (index, index) -> index
      "affine.yield"() : () -> ()
    }, {
      %2 = "add"(%1, %1) : (index, index) -> index
      "affine.yield"() : () -> ()
    }) {condition = #set1} : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "verbose_if"} : () -> ()
  "func.func"() ({
    "region"()[^bb1] ({
    }) : () -> ()
  ^bb1:  
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "terminator_with_regions"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    "unregistered_br"(%arg0)[^bb1] : (i1) -> ()
  ^bb1(%1: i1):  
    "func.return"(%1) : (i1) -> ()
  }) {function_type = (i1) -> i1, sym_name = "unregistered_term"} : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {dialect.attr = 10 : i64, function_type = () -> (), sym_name = "dialect_attrs"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "_valid.function$name", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {arg_attrs = [{}, {dialect.attr = 10 : i64}, {}], function_type = (i32, i1, i32) -> (), sym_name = "external_func_arg_attrs", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    "func.return"() : () -> ()
  }) {arg_attrs = [{dialect.attr = 10 : i64}], function_type = (i1) -> (), sym_name = "func_arg_attrs"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: f32):
    "func.return"(%arg0) : (f32) -> ()
  }) {function_type = (f32) -> f32, res_attrs = [{dialect.attr = 1 : i64}], sym_name = "func_result_attrs"} : () -> ()
  "func.func"() ({
  }) {function_type = (tuple<>) -> (), sym_name = "empty_tuple", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (tuple<i32>) -> (), sym_name = "tuple_single_element", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (tuple<i32, i16, f32>) -> (), sym_name = "tuple_multi_element", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (tuple<tuple<tuple<i32>>>) -> (), sym_name = "tuple_nested", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %1:2 = "foo_div"() : () -> (i16, i16)
    "func.return"(%1#0, %1#1) : (i16, i16) -> ()
  }) {function_type = () -> (i16, i16), sym_name = "pretty_form_multi_result"} : () -> ()
  "func.func"() ({
    %1:5 = "foo_test"() : () -> (i16, i16, i16, i16, i16)
    "func.return"(%1#0, %1#1, %1#2, %1#3, %1#4) : (i16, i16, i16, i16, i16) -> ()
  }) {function_type = () -> (i16, i16, i16, i16, i16), sym_name = "pretty_form_multi_result_groups"} : () -> ()
  "func.func"() ({
    "foo.unknown_op"() {foo = #foo.simple_attr} : () -> ()
    "foo.unknown_op"() {foo = #foo.complexattr<abcd>} : () -> ()
    "foo.unknown_op"() {foo = #foo.complexattr<abcd<f32>>} : () -> ()
    "foo.unknown_op"() {foo = #foo.complexattr<abcd<[f]$$[32]>>} : () -> ()
    "foo.unknown_op"() {foo = #foo.dialect<! × @#!@#>} : () -> ()
    "foo.unknown_op"() {foo = #foo<"dialect<! × @#!@#>>">} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "pretty_dialect_attribute"} : () -> ()
  "func.func"() ({
    %1 = "foo.unknown_op"() : () -> !foo.simpletype
    %2 = "foo.unknown_op"() : () -> !foo.complextype<abcd>
    %3 = "foo.unknown_op"() : () -> !foo.complextype<abcd<f32>>
    %4 = "foo.unknown_op"() : () -> !foo.complextype<abcd<[f]$$[32]>>
    %5 = "foo.unknown_op"() : () -> !foo.dialect<! × @#!@#>
    %6 = "foo.unknown_op"() : () -> !foo<"dialect<! × @#!@#>>">
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "pretty_dialect_type"} : () -> ()
  "func.func"() ({
    %1 = "foo.unknown_op"() : () -> none
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "none_type"} : () -> ()
  "func.func"() ({
    "foo.region_op"() ({
      %1 = "foo.unknown_op"() : () -> none
      "foo.terminator"() : () -> ()
    }, {
      %1 = "foo.unknown_op"() : () -> none
      "foo.terminator"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "scoped_names"} : () -> ()
  "func.func"() ({
    "foo.unknown_op"() {foo = #foo.attr : i32} : () -> ()
  }) {function_type = () -> (), sym_name = "dialect_attribute_with_type"} : () -> ()
  "func.func"() ({
    %1 = "arith.constant"() {value = 0x7C01 : f16} : () -> f16
    %2 = "arith.constant"() {value = 0x7FFF : f16} : () -> f16
    %3 = "arith.constant"() {value = 0xFFFF : f16} : () -> f16
    %4 = "arith.constant"() {value = 0x7C00 : f16} : () -> f16
    %5 = "arith.constant"() {value = 0xFC00 : f16} : () -> f16
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "f16_special_values"} : () -> ()
  "func.func"() ({
    %1 = "arith.constant"() {value = 0x7F800001 : f32} : () -> f32
    %2 = "arith.constant"() {value = 0x7FBFFFFF : f32} : () -> f32
    %3 = "arith.constant"() {value = 0x7FC00000 : f32} : () -> f32
    %4 = "arith.constant"() {value = 0xFFFFFFFF : f32} : () -> f32
    %5 = "arith.constant"() {value = 0x7F800000 : f32} : () -> f32
    %6 = "arith.constant"() {value = 0xFF800000 : f32} : () -> f32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "f32_special_values"} : () -> ()
  "func.func"() ({
    %1 = "arith.constant"() {value = 0x7FF0000000000001 : f64} : () -> f64
    %2 = "arith.constant"() {value = 0x7FF8000000000000 : f64} : () -> f64
    %3 = "arith.constant"() {value = 0x7FF0000001000000 : f64} : () -> f64
    %4 = "arith.constant"() {value = 0xFFF0000001000000 : f64} : () -> f64
    %5 = "arith.constant"() {value = 0x7FF0000000000000 : f64} : () -> f64
    %6 = "arith.constant"() {value = 0xFFF0000000000000 : f64} : () -> f64
    %7 = "arith.constant"() {value = 0xC1CDC00000000000 : f64} : () -> f64
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "f64_special_values"} : () -> ()
  "func.func"() ({
    %1 = "arith.constant"() {value = 0x7F81 : bf16} : () -> bf16
    %2 = "arith.constant"() {value = 0xFF81 : bf16} : () -> bf16
    %3 = "arith.constant"() {value = 0x7FC0 : bf16} : () -> bf16
    %4 = "arith.constant"() {value = 0xFFC0 : bf16} : () -> bf16
    %5 = "arith.constant"() {value = 0x7F80 : bf16} : () -> bf16
    %6 = "arith.constant"() {value = 0xFF80 : bf16} : () -> bf16
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "bfloat16_special_values"} : () -> ()
  "func.func"() ({
    %1 = "arith.constant"() {value = -1.23697901 : f32} : () -> f32
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "f32_potential_precision_loss"} : () -> ()
  "func.func"() ({
    "foo"() {bar = dense<0 × FFFFFFFF> : tensor<4 × 4 × f32>} : () -> ()
    "foo"() {bar = dense<[[0 × FFFFFFFF, 0 × 7F800000], [0 × 7FBFFFFF, 0 × 7F800001]]> : tensor<2 × 2 × f32>} : () -> ()
    "foo"() {bar = dense<[0 × FFFFFFFF, 0.00000000]> : tensor<2 × f32>} : () -> ()
    "foo"() {bar = sparse<[[1, 1, 0], [0, 1, 1]], [0 × FFFFFFFF, 0 × 7F800001]> : tensor<2 × 2 × 2 × f32>} : () -> ()
  }) {function_type = () -> (), sym_name = "special_float_values_in_tensors"} : () -> ()
  "func.func"() ({
    "test.polyfor"() ({
    ^bb0(%arg0: index, %arg1: index, %arg2: index):
      "foo"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "op_with_region_args"} : () -> ()
  "func.func"() ({
    %1 = "arith.constant"() {value = 10 : index} : () -> index
    "test.isolated_region"(%1) ({
    ^bb0(%arg0: index):
      "foo.consumer"(%arg0) : (index) -> ()
    }) : (index) -> ()
    %2:2 = "foo.op"() : () -> (index, index)
    "test.isolated_region"(%2#1) ({
    ^bb0(%arg0: index):
      "foo.consumer"(%arg0) : (index) -> ()
    }) : (index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "op_with_passthrough_region_args"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> !unreg.ptr<() -> ()>, sym_name = "ptr_to_function", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {arg_attrs = [{foo.value = "\0A"}], function_type = (i1) -> (), sym_name = "escaped_string_char", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %1:5 = "test.parse_integer_literal"() : () -> (index, index, index, index, index)
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "parse_integer_literal_test"} : () -> ()
  "func.func"() ({
    "test.parse_wrapped_keyword"() {keyword = "foo.keyword"} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "parse_wrapped_keyword_test"} : () -> ()
  "func.func"() ({
    "foo.symbol_reference"() {ref = @"\22_string_symbol_reference\22"} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "\22_string_symbol_reference\22"} : () -> ()
  "func.func"() ({
    "foo.constant"() {value = #foo<"\22escaped\\\0A\22">} : () -> ()
  }) {function_type = () -> (), sym_name = "parse_opaque_attr_escape", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {"0 . 0", function_type = () -> (), nested = {"0 . 0"}, sym_name = "string_attr_name", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "nested_reference", sym_visibility = "private", test.ref = @some_symbol::@some_nested_symbol} : () -> ()
  "func.func"() ({
    %1:4 = "test.asm_interface_op"() : () -> (i32, i32, i32, i32)
    %2:2 = "test.asm_interface_op"() : () -> (i32, i32)
    "func.return"(%1#0, %1#1, %1#2, %1#3, %2#0, %2#1) : (i32, i32, i32, i32, i32, i32) -> ()
  }) {function_type = () -> (i32, i32, i32, i32, i32, i32), sym_name = "custom_asm_names"} : () -> ()
  "func.func"() ({
    %1 = "test.string_attr_pretty_name"() {names = ["x"]} : () -> i32
    %2 = "test.string_attr_pretty_name"() {names = ["y"]} : () -> i32
    %3 = "test.string_attr_pretty_name"() {names = ["y"]} : () -> i32
    %4 = "test.string_attr_pretty_name"() {names = ["space name"]} : () -> i32
    "unknown.use"(%1, %2, %3, %4) : (i32, i32, i32, i32) -> ()
    %5:3 = "test.string_attr_pretty_name"() {names = ["a", "b", "c"]} : () -> (i32, i32, i32)
    %6:4 = "test.string_attr_pretty_name"() {names = ["q", "q", "q", "r"]} : () -> (i32, i32, i32, i32)
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "pretty_names"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i1):
    "test.default_dialect"() ({
      %1:5 = "test.parse_integer_literal"() : () -> (index, index, index, index, index)
      %2:6 = "test.parse_integer_literal"() : () -> (index, index, index, index, index, index)
      "test.op_with_attr"() {test.attr = "test.value"} : () -> ()
      "test.terminator"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = (i1) -> (), sym_name = "default_dialect"} : () -> ()
  "func.func"() ({
    %1 = "arith.constant"() {value = false} : () -> i1
    "func.return"(%1) : (i1) -> ()
  ^bb1:  
    %2:3 = "bar"(%3) : (i64) -> (i1, i1, i1)
    "cf.br"()[^bb3] : () -> ()
  ^bb2:  
    "cf.br"()[^bb2] : () -> ()
  ^bb3:  
    %3 = "foo"() : () -> i64
    "func.return"(%2#1) : (i1) -> ()
  }) {function_type = () -> i1, sym_name = "unreachable_dominance_violation_ok"} : () -> ()
  "func.func"() ({
    "cf.br"()[^bb2] : () -> ()
  ^bb1:  
    "test.graph_region"() ({
      %2:3 = "bar"(%1) : (i64) -> (i1, i1, i1)
    }) : () -> ()
    "cf.br"()[^bb3] : () -> ()
  ^bb2:  
    %1 = "foo"() : () -> i64
    "cf.br"()[^bb1] : () -> ()
  ^bb3:  
    "func.return"(%1) : (i64) -> ()
  }) {function_type = () -> i64, sym_name = "graph_region_in_hierarchy_ok"} : () -> ()
  "func.func"() ({
    "test.graph_region"() ({
      %1:3 = "bar"(%2) : (i64) -> (i1, i1, i1)
      %2 = "baz"(%1#0) : (i1) -> i64
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "graph_region_kind"} : () -> ()
  "func.func"() ({
    "test.ssacfg_region"() ({
      %1 = "baz"() : () -> i64
      "test.graph_region"() ({
        %3:3 = "bar"(%1) : (i64) -> (i1, i1, i1)
      }) : () -> ()
      %2 = "baz"() : () -> i64
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "graph_region_inside_ssacfg_region"} : () -> ()
  "func.func"() ({
    "test.graph_region"() ({
      "test.graph_region"() ({
        %2:3 = "bar"(%1) : (i64) -> (i1, i1, i1)
      }) : () -> ()
      %1 = "foo"() : () -> i64
      "test.terminator"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "graph_region_in_graph_region_ok"} : () -> ()
  "test.graph_region"() ({
    %1 = "op1"(%3) : (i32) -> i32
    %2 = "test.ssacfg_region"(%1, %2, %3, %4) ({
      %5 = "op2"(%1, %2, %3, %4) : (i32, i32, i32, i32) -> i32
    }) : (i32, i32, i32, i32) -> i32
    %3 = "op2"(%1, %4) : (i32, i32) -> i32
    %4 = "op3"(%1) : (i32) -> i32
  }) : () -> ()
  "unregistered_func_might_have_graph_region"() ({
    %1 = "foo"(%1, %2) : (i64, i64) -> i64
    %2 = "bar"(%1) : (i64) -> i64
    "unregistered_terminator"() : () -> ()
  }) {function_type = () -> i1, sym_name = "unregistered_op_dominance_violation_ok"} : () -> ()
  "test.dialect_custom_printer"() : () -> ()
  "test.dialect_custom_format_fallback"() : () -> ()
  %0 = "test.format_optional_result_d_op"() : () -> f80
}) : () -> ()


