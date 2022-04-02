
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
  ^bb0(%arg0: memref<* × i32>, %arg1: tensor<* × i32>):
    "memref.tensor_store"(%arg1, %arg0) : (tensor<* × i32>, memref<* × i32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<* × i32>, tensor<* × i32>) -> (), sym_name = "unranked_tensor_load_store"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZcore-ops.out.txt" str
