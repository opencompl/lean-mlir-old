"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<? × ? × f32>):
    %0 = "arith.constant"() {value = 2 : index} : () -> index
    %1 = "arith.constant"() {value = 44 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    "scf.for"(%0, %1, %2) ({
    ^bb0(%arg1: index):
      "scf.for"(%2, %1, %0) ({
      ^bb0(%arg2: index):
        %3 = "memref.load"(%arg0, %arg1, %arg2) : (memref<? × ? × f32>, index, index) -> f32
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<? × ? × f32>) -> (), sym_name = "rectangular"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<? × ? × f32>):
    %0 = "arith.constant"() {value = 2 : index} : () -> index
    %1 = "arith.constant"() {value = 44 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    "scf.for"(%0, %1, %2) ({
    ^bb0(%arg1: index):
      "scf.for"(%2, %arg1, %0) ({
      ^bb0(%arg2: index):
        %3 = "memref.load"(%arg0, %arg1, %arg2) : (memref<? × ? × f32>, index, index) -> f32
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<? × ? × f32>) -> (), sym_name = "triangular"} : () -> ()
}) : () -> ()

// -----
