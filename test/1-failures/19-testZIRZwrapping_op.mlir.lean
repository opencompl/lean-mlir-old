
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
  ^bb0(%arg0: i32, %arg1: f32):
    %0:3 = "test.wrapping_region"() ({
      %1:3 = "some.op"(%arg1, %arg0) {test.attr = "attr"} : (f32, i32) -> (i1, i2, i3)
      "test.return"(%1#0, %1#1, %1#2) : (i1, i2, i3) -> ()
    }) : () -> (i1, i2, i3)
    "func.return"(%0#2, %0#1, %0#0) : (i3, i2, i1) -> ()
  }) {function_type = (i32, f32) -> (i3, i2, i1), sym_name = "wrapping_op"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZwrapping_op.out.txt" str
