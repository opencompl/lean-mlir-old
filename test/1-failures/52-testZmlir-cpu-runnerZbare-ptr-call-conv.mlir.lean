
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
  "func.func"() ({
  ^bb0(%arg0: memref<2 × f32>, %arg1: memref<2 × f32>):
    %0 = "arith.constant"() {value = 2 : index} : () -> index
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "arith.constant"() {value = 1.00000000 : f32} : () -> f32
    %4 = "arith.constant"() {value = 2.00000000 : f32} : () -> f32
    "scf.for"(%1, %0, %2) ({
    ^bb0(%arg2: index):
      %5 = "memref.load"(%arg0, %arg2) : (memref<2 × f32>, index) -> f32
      %6 = "arith.addf"(%5, %3) : (f32, f32) -> f32
      "memref.store"(%6, %arg0, %arg2) : (f32, memref<2 × f32>, index) -> ()
      %7 = "memref.load"(%arg1, %arg2) : (memref<2 × f32>, index) -> f32
      %8 = "arith.addf"(%6, %4) : (f32, f32) -> f32
      "memref.store"(%8, %arg1, %arg2) : (f32, memref<2 × f32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<2 × f32>, memref<2 × f32>) -> (), sym_name = "simple_add1_add2_test"} : () -> ()
  "llvm.func"() ({
  }) {function_type = !llvm.func<ptr<i8> (i64)>, linkage = #llvm.linkage<e × ternal>, sym_name = "malloc"} : () -> ()
  "llvm.func"() ({
  }) {function_type = !llvm.func<void (ptr<i8>)>, linkage = #llvm.linkage<e × ternal>, sym_name = "free"} : () -> ()
  "func.func"() ({
  }) {function_type = (f32) -> (), sym_name = "printF32", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "printComma", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "printNewline", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 2 : index} : () -> index
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "arith.constant"() {value = 1.00000000 : f32} : () -> f32
    %4 = "arith.constant"() {value = 2.00000000 : f32} : () -> f32
    %5 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    %6 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2 × i32>} : () -> memref<2 × f32>
    "scf.for"(%1, %0, %2) ({
    ^bb0(%arg0: index):
      "memref.store"(%3, %5, %arg0) : (f32, memref<2 × f32>, index) -> ()
      "memref.store"(%3, %6, %arg0) : (f32, memref<2 × f32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.call"(%5, %6) {callee = @simple_add1_add2_test} : (memref<2 × f32>, memref<2 × f32>) -> ()
    %7 = "memref.load"(%5, %1) : (memref<2 × f32>, index) -> f32
    "func.call"(%7) {callee = @printF32} : (f32) -> ()
    "func.call"() {callee = @printComma} : () -> ()
    %8 = "memref.load"(%5, %2) : (memref<2 × f32>, index) -> f32
    "func.call"(%8) {callee = @printF32} : (f32) -> ()
    "func.call"() {callee = @printNewline} : () -> ()
    %9 = "memref.load"(%6, %1) : (memref<2 × f32>, index) -> f32
    "func.call"(%9) {callee = @printF32} : (f32) -> ()
    "func.call"() {callee = @printComma} : () -> ()
    %10 = "memref.load"(%6, %2) : (memref<2 × f32>, index) -> f32
    "func.call"(%10) {callee = @printF32} : (f32) -> ()
    "func.call"() {callee = @printNewline} : () -> ()
    "memref.dealloc"(%5) : (memref<2 × f32>) -> ()
    "memref.dealloc"(%6) : (memref<2 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZmlir-cpu-runnerZbare-ptr-call-conv.out.txt" str
