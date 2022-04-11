
import MLIR.Doc
import MLIR.AST
import MLIR.EDSL

open Lean
open Lean.Parser
open  MLIR.EDSL
open MLIR.AST
open MLIR.Doc
open IO

set_option maxHeartbeats 999999999


declare_syntax_cat mlir_ops
syntax (ws mlir_op ws)* : mlir_ops
syntax "[mlir_ops|" mlir_ops "]" : term

macro_rules
|`([mlir_ops| $[ $xs ]* ]) => do 
  let xs <- xs.mapM (fun x =>`([mlir_op| $x]))
  quoteMList xs.toList (<-`(MLIR.AST.Op))

  
-- | write an op into the path
def o: List Op := [mlir_ops|
"builtin.module"() ({
  "llvm.mlir.global"() ({
  }) {constant, global_type = !llvm.array<16  ×  i8>, linkage = #llvm.linkage<internal>, sym_name = "str_global", unnamed_addr = 0 : i64, value = "String to print\0A"} : () -> ()
  "llvm.func"() ({
  }) {function_type = !llvm.func<void (ptr<i8>)>, linkage = #llvm.linkage<e × ternal>, sym_name = "print_c_string"} : () -> ()
  "func.func"() ({
    %0 = "llvm.mlir.addressof"() {global_name = @str_global} : () -> !llvm.ptr<array<16  ×  i8>>
    %1 = "llvm.mlir.constant"() {value = 0 : index} : () -> i64
    %2 = "llvm.getelementptr"(%0, %1, %1) {structIndices = dense<-2147483648> : tensor<2 × i32>} : (!llvm.ptr<array<16  ×  i8>>, i64, i64) -> !llvm.ptr<i8>
    "llvm.call"(%2) {callee = @print_c_string} : (!llvm.ptr<i8>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZmlir-cpu-runnerZprint.out.txt" str
