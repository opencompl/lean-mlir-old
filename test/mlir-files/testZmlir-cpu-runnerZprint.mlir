"builtin.module"() ({
  "llvm.mlir.global"() ({
  }) {constant, global_type = !llvm.array<16 x i8>, linkage = #llvm.linkage<internal>, sym_name = "str_global", unnamed_addr = 0 : i64, value = "String to print\0A"} : () -> ()
  "llvm.func"() ({
  }) {function_type = !llvm.func<void (ptr<i8>)>, linkage = #llvm.linkage<external>, sym_name = "print_c_string"} : () -> ()
  "func.func"() ({
    %0 = "llvm.mlir.addressof"() {global_name = @str_global} : () -> !llvm.ptr<array<16 x i8>>
    %1 = "llvm.mlir.constant"() {value = 0 : index} : () -> i64
    %2 = "llvm.getelementptr"(%0, %1, %1) {structIndices = dense<-2147483648> : tensor<2xi32>} : (!llvm.ptr<array<16 x i8>>, i64, i64) -> !llvm.ptr<i8>
    "llvm.call"(%2) {callee = @print_c_string} : (!llvm.ptr<i8>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()

// -----
