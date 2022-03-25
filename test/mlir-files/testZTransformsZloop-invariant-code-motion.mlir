#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
    %2 = "arith.constant"() {value = 8.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %3 = "arith.addf"(%1, %2) : (f32, f32) -> f32
      "affine.for"() ({
      ^bb0(%arg1: index):
        %4 = "arith.addf"(%3, %2) : (f32, f32) -> f32
        "affine.store"(%3, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "nested_loops_both_having_invariant_code"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<() -> (0)>
#map1 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
    %2 = "arith.constant"() {value = 8.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        %3 = "arith.addf"(%1, %2) : (f32, f32) -> f32
        "affine.yield"() : () -> ()
      }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "nested_loops_code_invariant_to_both"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.load"(%0, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      %3 = "affine.load"(%1, %arg0) {map = #map0} : (memref<10 × f32>, index) -> f32
      %4 = "arith.addf"(%2, %3) : (f32, f32) -> f32
      "affine.store"(%4, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "single_loop_nothing_invariant"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (10)>
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 8.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %2 = "affine.apply"(%arg0) {map = #map0} : (index) -> index
      "affine.if"(%arg0, %2) ({
        %3 = "arith.addf"(%1, %1) : (f32, f32) -> f32
        "affine.store"(%3, %0, %arg0) {map = #map1} : (f32, memref<10 × f32>, index) -> ()
        "affine.yield"() : () -> ()
      }, {
      }) {condition = #set} : (index, index) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map2, step = 1 : index, upper_bound = #map3} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "invariant_code_inside_affine_if"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<() -> (0)>
#map1 = affine_map<() -> (10)>
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 8.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.if"(%arg0, %arg0) ({
          %2 = "arith.addf"(%1, %1) : (f32, f32) -> f32
          "affine.yield"() : () -> ()
        }, {
        }) {condition = #set} : (index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "invariant_affine_if"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 8.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.if"(%arg0, %arg0) ({
          %2 = "arith.addf"(%1, %1) : (f32, f32) -> f32
          "affine.store"(%2, %0, %arg1) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
          "affine.yield"() : () -> ()
        }, {
        }) {condition = #set} : (index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "invariant_affine_if2"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<() -> (0)>
#map1 = affine_map<() -> (10)>
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 8.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.if"(%arg0, %arg0) ({
          %2 = "arith.addf"(%1, %1) : (f32, f32) -> f32
          "affine.if"(%arg0, %arg0) ({
            %3 = "arith.addf"(%2, %2) : (f32, f32) -> f32
            "affine.yield"() : () -> ()
          }, {
          }) {condition = #set} : (index, index) -> ()
          "affine.yield"() : () -> ()
        }, {
        }) {condition = #set} : (index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "invariant_affine_nested_if"} : () -> ()
}) : () -> ()

// -----
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (10)>
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %1 = "arith.constant"() {value = 8.000000e+00 : f32} : () -> f32
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.for"() ({
      ^bb0(%arg1: index):
        "affine.if"(%arg0, %arg0) ({
          %2 = "arith.addf"(%1, %1) : (f32, f32) -> f32
          "affine.store"(%2, %0, %arg0) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
          "affine.if"(%arg0, %arg0) ({
            %3 = "arith.addf"(%2, %2) : (f32, f32) -> f32
            "affine.yield"() : () -> ()
          }, {
            "affine.store"(%2, %0, %arg1) {map = #map0} : (f32, memref<10 × f32>, index) -> ()
            "affine.yield"() : () -> ()
          }) {condition = #set} : (index, index) -> ()
          "affine.yield"() : () -> ()
        }, {
        }) {condition = #set} : (index, index) -> ()
        "affine.yield"() : () -> ()
      }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "invariant_affine_nested_if_else"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 10 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    %4 = "arith.constant"() {value = 7.000000e+00 : f32} : () -> f32
    %5 = "arith.constant"() {value = 8.000000e+00 : f32} : () -> f32
    "scf.for"(%0, %1, %2) ({
    ^bb0(%arg0: index):
      "scf.for"(%0, %1, %2) ({
      ^bb0(%arg1: index):
        %6 = "arith.addf"(%4, %5) : (f32, f32) -> f32
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "invariant_loop_dialect"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 10 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<10 × f32>
    "scf.for"(%0, %1, %2) ({
    ^bb0(%arg0: index):
      "scf.for"(%0, %1, %2) ({
      ^bb0(%arg1: index):
        %4 = "arith.addi"(%arg0, %arg1) : (index, index) -> index
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "variant_loop_dialect"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 10 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "arith.constant"() {value = 7 : i32} : () -> i32
    %4 = "arith.constant"() {value = 8 : i32} : () -> i32
    "scf.parallel"(%0, %0, %1, %1, %2, %2) ({
    ^bb0(%arg0: index, %arg1: index):
      %5 = "arith.addi"(%3, %4) : (i32, i32) -> i32
      %6 = "arith.addi"(%arg0, %arg1) : (index, index) -> index
      "scf.yield"() : () -> ()
    }) {operand_segment_sizes = dense<[2, 2, 2, 0]> : vector<4 × i32>} : (index, index, index, index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "parallel_loop_with_invariant"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> index, sym_name = "make_val", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    %0 = "arith.constant"() {value = true} : () -> i1
    "scf.for"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: index):
      %1 = "func.call"() {callee = @make_val} : () -> index
      %2 = "func.call"() {callee = @make_val} : () -> index
      %3 = "scf.if"(%0) ({
        "scf.yield"(%1) : (index) -> ()
      }, {
        "scf.yield"(%2) : (index) -> ()
      }) : (i1) -> index
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index) -> (), sym_name = "nested_uses_inside"} : () -> ()
}) : () -> ()

// -----
