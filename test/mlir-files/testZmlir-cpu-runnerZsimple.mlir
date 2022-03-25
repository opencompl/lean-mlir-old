"builtin.module"() ({
  "llvm.func"() ({
  }) {function_type = !llvm.func<f32 (f32)>, linkage = #llvm.linkage<external>, sym_name = "fabsf"} : () -> ()
  "llvm.func"() ({
  }) {function_type = !llvm.func<ptr<i8> (i64)>, linkage = #llvm.linkage<external>, sym_name = "malloc"} : () -> ()
  "llvm.func"() ({
  }) {function_type = !llvm.func<void (ptr<i8>)>, linkage = #llvm.linkage<external>, sym_name = "free"} : () -> ()
  "llvm.func"() ({
    %0 = "llvm.mlir.constant"() {value = -4.200000e+02 : f32} : () -> f32
    %1 = "llvm.call"(%0) {callee = @fabsf} : (f32) -> f32
    "llvm.return"(%1) : (f32) -> ()
  }) {function_type = !llvm.func<f32 ()>, linkage = #llvm.linkage<external>, sym_name = "main"} : () -> ()
  "llvm.func"() ({
    %0 = "llvm.mlir.constant"() {value = 4 : index} : () -> i64
    %1 = "llvm.call"(%0) {callee = @malloc} : (i64) -> !llvm.ptr<i8>
    %2 = "llvm.bitcast"(%1) : (!llvm.ptr<i8>) -> !llvm.ptr<f32>
    "llvm.return"(%2) : (!llvm.ptr<f32>) -> ()
  }) {function_type = !llvm.func<ptr<f32> ()>, linkage = #llvm.linkage<external>, sym_name = "allocation"} : () -> ()
  "llvm.func"() ({
  ^bb0(%arg0: !llvm.ptr<f32>):
    %0 = "llvm.bitcast"(%arg0) : (!llvm.ptr<f32>) -> !llvm.ptr<i8>
    "llvm.call"(%0) {callee = @free} : (!llvm.ptr<i8>) -> ()
    "llvm.return"() : () -> ()
  }) {function_type = !llvm.func<void (ptr<f32>)>, linkage = #llvm.linkage<external>, sym_name = "deallocation"} : () -> ()
  "llvm.func"() ({
    %0 = "llvm.call"() {callee = @allocation} : () -> !llvm.ptr<f32>
    %1 = "llvm.mlir.constant"() {value = 0 : index} : () -> i64
    %2 = "llvm.mlir.constant"() {value = 1.234000e+03 : f32} : () -> f32
    %3 = "llvm.getelementptr"(%0, %1) {structIndices = dense<-2147483648> : tensor<1xi32>} : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    "llvm.store"(%2, %3) : (f32, !llvm.ptr<f32>) -> ()
    %4 = "llvm.getelementptr"(%0, %1) {structIndices = dense<-2147483648> : tensor<1xi32>} : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %5 = "llvm.load"(%4) : (!llvm.ptr<f32>) -> f32
    "llvm.call"(%0) {callee = @deallocation} : (!llvm.ptr<f32>) -> ()
    "llvm.return"(%5) : (f32) -> ()
  }) {function_type = !llvm.func<f32 ()>, linkage = #llvm.linkage<external>, sym_name = "foo"} : () -> ()
  "llvm.func"() ({
    %0 = "llvm.mlir.constant"() {value = 42 : i32} : () -> i32
    "llvm.return"(%0) : (i32) -> ()
  }) {function_type = !llvm.func<i32 ()>, linkage = #llvm.linkage<external>, sym_name = "int32_main"} : () -> ()
  "llvm.func"() ({
    %0 = "llvm.mlir.constant"() {value = 42 : i64} : () -> i64
    "llvm.return"(%0) : (i64) -> ()
  }) {function_type = !llvm.func<i64 ()>, linkage = #llvm.linkage<external>, sym_name = "int64_main"} : () -> ()
}) : () -> ()

// -----
