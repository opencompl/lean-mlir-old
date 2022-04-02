
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
  ^bb0(%arg0: tensor<4 × 4 × ? × f32>, %arg1: f32, %arg2: i32, %arg3: index, %arg4: i64, %arg5: f16):
    %19 = "arith.constant"() {value = dense<0.00000000> : vector<4 × f32>} : () -> vector<4 × f32>
  }) {function_type = (tensor<4 × 4 × ? × f32>, f32, i32, index, i64, f16) -> (), sym_name = "standard_instrs"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZcore-ops.out.txt" str
