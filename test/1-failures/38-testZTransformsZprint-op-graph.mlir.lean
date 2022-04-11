
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
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "arith.constant"() {value = dense<[[0, 1], [2, 3]]> : tensor<2 × 2 × i32>} : () -> tensor<2 × 2 × i32>
    %1 = "arith.constant"() {value = dense<1> : tensor<5 × i32>} : () -> tensor<5 × i32>
    %2 = "arith.constant"() {value = dense<[[0, 1]]> : tensor<1 × 2 × i32>} : () -> tensor<1 × 2 × i32>
    %3 = "arith.constant"() {value = 10 : i32} : () -> i32
    %4 = "test.func"() : () -> i32
    %5:2 = "test.merge_blocks"() ({
      "test.br"(%arg0, %4, %3)[^bb1] : (i32, i32, i32) -> ()
    ^bb1(%6: i32, %7: i32, %8: i32):  
      "test.return"(%6, %7) : (i32, i32) -> ()
    }) : () -> (i32, i32)
    "test.return"(%5#0, %5#1) : (i32, i32) -> ()
  }) {function_type = (i32, i32) -> (), sym_name = "merge_blocks"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZprint-op-graph.out.txt" str
