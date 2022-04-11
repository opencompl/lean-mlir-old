
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
^bb0:
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (f32) -> (), sym_name = "t", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {arg_attrs = [{test.A}, {test.B}], function_type = (f32, f32) -> (), sym_name = "erase_arg", sym_visibility = "private", test.set_type_from = @t} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> f32, sym_name = "t", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (f32, f32), res_attrs = [{test.A}, {test.B}], sym_name = "erase_result", sym_visibility = "private", test.set_type_from = @t} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZtest-func-set-type.out.txt" str
