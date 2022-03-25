#map0 = affine_map<(d0) -> (d0 mod 2)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (4)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.constant"() {value = 1 : i32} : () -> i32
    "func.return"(%0, %1) : (i32, i32) -> ()
  }) {function_type = () -> (i32, i32), sym_name = "simple_constant"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "affine.apply"(%0) {map = #map0} : (index) -> index
    %3 = "affine.apply"(%1) {map = #map0} : (index) -> index
    "func.return"(%2, %3) : (index, index) -> ()
  }) {function_type = () -> (index, index), sym_name = "basic"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = "arith.addf"(%arg0, %arg1) : (f32, f32) -> f32
    %1 = "arith.addf"(%arg0, %arg1) : (f32, f32) -> f32
    %2 = "arith.addf"(%arg0, %arg1) : (f32, f32) -> f32
    %3 = "arith.addf"(%arg0, %arg1) : (f32, f32) -> f32
    %4 = "arith.addf"(%0, %1) : (f32, f32) -> f32
    %5 = "arith.addf"(%2, %3) : (f32, f32) -> f32
    %6 = "arith.addf"(%0, %2) : (f32, f32) -> f32
    %7 = "arith.addf"(%4, %5) : (f32, f32) -> f32
    %8 = "arith.addf"(%5, %6) : (f32, f32) -> f32
    %9 = "arith.addf"(%7, %8) : (f32, f32) -> f32
    "func.return"(%9) : (f32) -> ()
  }) {function_type = (f32, f32) -> f32, sym_name = "many"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.constant"() {value = 1 : i32} : () -> i32
    "func.return"(%0, %1) : (i32, i32) -> ()
  }) {function_type = () -> (i32, i32), sym_name = "different_ops"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<*xf32>):
    %0 = "tensor.cast"(%arg0) : (tensor<*xf32>) -> tensor<?x?xf32>
    %1 = "tensor.cast"(%arg0) : (tensor<*xf32>) -> tensor<4x?xf32>
    "func.return"(%0, %1) : (tensor<?x?xf32>, tensor<4x?xf32>) -> ()
  }) {function_type = (tensor<*xf32>) -> (tensor<?x?xf32>, tensor<4x?xf32>), sym_name = "different_results"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "arith.cmpi"(%arg0, %arg1) {predicate = 2 : i64} : (index, index) -> i1
    %1 = "arith.cmpi"(%arg0, %arg1) {predicate = 1 : i64} : (index, index) -> i1
    %2 = "arith.cmpi"(%arg0, %arg1) {predicate = 1 : i64} : (index, index) -> i1
    "func.return"(%0, %1, %2) : (i1, i1, i1) -> ()
  }) {function_type = (index, index) -> (i1, i1, i1), sym_name = "different_attributes"} : () -> ()
  "func.func"() ({
    %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<2x1xf32>
    %1 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2xi32>} : () -> memref<2x1xf32>
    "func.return"(%0, %1) : (memref<2x1xf32>, memref<2x1xf32>) -> ()
  }) {function_type = () -> (memref<2x1xf32>, memref<2x1xf32>), sym_name = "side_effect"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    "affine.for"() ({
    ^bb0(%arg0: index):
      %1 = "arith.constant"() {value = 1 : i32} : () -> i32
      "foo"(%0, %1) : (i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "down_propagate_for"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.constant"() {value = true} : () -> i1
    "cf.cond_br"(%1, %0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : (i1, i32) -> ()
  ^bb1:  // pred: ^bb0
    %2 = "arith.constant"() {value = 1 : i32} : () -> i32
    "cf.br"(%2)[^bb2] : (i32) -> ()
  ^bb2(%3: i32):  // 2 preds: ^bb0, ^bb1
    "func.return"(%3) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "down_propagate"} : () -> ()
  "func.func"() ({
    "affine.for"() ({
    ^bb0(%arg0: index):
      %1 = "arith.constant"() {value = 1 : i32} : () -> i32
      "foo"(%1) : (i32) -> ()
      "affine.yield"() : () -> ()
    }) {lower_bound = #map1, step = 1 : index, upper_bound = #map2} : () -> ()
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "up_propagate_for"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : i32} : () -> i32
    %1 = "arith.constant"() {value = true} : () -> i1
    "cf.cond_br"(%1, %0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : (i1, i32) -> ()
  ^bb1:  // pred: ^bb0
    %2 = "arith.constant"() {value = 1 : i32} : () -> i32
    "cf.br"(%2)[^bb2] : (i32) -> ()
  ^bb2(%3: i32):  // 2 preds: ^bb0, ^bb1
    %4 = "arith.constant"() {value = 1 : i32} : () -> i32
    %5 = "arith.addi"(%3, %4) : (i32, i32) -> i32
    "func.return"(%5) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "up_propagate"} : () -> ()
  "func.func"() ({
    %0 = "foo.region"() ({
      %1 = "arith.constant"() {value = 0 : i32} : () -> i32
      %2 = "arith.constant"() {value = true} : () -> i1
      "cf.cond_br"(%2, %1)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : (i1, i32) -> ()
    ^bb1:  // pred: ^bb0
      %3 = "arith.constant"() {value = 1 : i32} : () -> i32
      "cf.br"(%3)[^bb2] : (i32) -> ()
    ^bb2(%4: i32):  // 2 preds: ^bb0, ^bb1
      %5 = "arith.constant"() {value = 1 : i32} : () -> i32
      %6 = "arith.addi"(%4, %5) : (i32, i32) -> i32
      "foo.yield"(%6) : (i32) -> ()
    }) : () -> i32
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "up_propagate_region"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    "func.func"() ({
      %1 = "arith.constant"() {value = 1 : i32} : () -> i32
      "foo.yield"(%1) : (i32) -> ()
    }) {function_type = () -> (), sym_name = "nested_func"} : () -> ()
    "foo.region"() ({
      %1 = "arith.constant"() {value = 1 : i32} : () -> i32
      "foo.yield"(%1) : (i32) -> ()
    }) : () -> ()
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "nested_isolated"} : () -> ()
  "func.func"() ({
    "test.graph_region"() ({
      %0 = "arith.addi"(%1, %2) : (i32, i32) -> i32
      %1 = "arith.constant"() {value = 1 : i32} : () -> i32
      %2 = "arith.constant"() {value = 1 : i32} : () -> i32
      "foo.yield"(%0) : (i32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "use_before_def"} : () -> ()
}) : () -> ()

// -----
