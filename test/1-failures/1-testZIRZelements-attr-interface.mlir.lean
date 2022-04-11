
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
  %0 = "arith.constant"() {value = #test.i64_elements<[10, 11, 12, 13, 14] : tensor<5 × i64>>} : () -> tensor<5 × i64>
  %1 = "arith.constant"() {value = dense<[10, 11, 12, 13, 14]> : tensor<5 × i64>} : () -> tensor<5 × i64>
  %2 = "arith.constant"() {value = opaque<"_", "0 × DEADBEEF"> : tensor<5 × i64>} : () -> tensor<5 × i64>
  %3 = "arith.constant"() {value = dense<> : tensor<0 × i64>} : () -> tensor<0 × i64>
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZelements-attr-interface.out.txt" str
