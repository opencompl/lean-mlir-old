
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
  }) {function_type = () -> f32, res_attrs = [{test.erase_this_result}], sym_name = "f", sym_visibility = "private"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (f32, f32), res_attrs = [{test.erase_this_result}, {test.A}], sym_name = "f", sym_visibility = "private"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (f32, f32), res_attrs = [{test.A}, {test.erase_this_result}], sym_name = "f", sym_visibility = "private"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (f32, f32, f32), res_attrs = [{test.A}, {test.erase_this_result}, {test.B}], sym_name = "f", sym_visibility = "private"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (f32, f32, f32, f32), res_attrs = [{test.A}, {test.erase_this_result}, {test.erase_this_result}, {test.B}], sym_name = "f", sym_visibility = "private"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (f32, f32, f32, f32, f32), res_attrs = [{test.A}, {test.erase_this_result}, {test.B}, {test.erase_this_result}, {test.C}], sym_name = "f", sym_visibility = "private"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (tensor<1 × f32>, f32, tensor<2 × f32>, f32, tensor<3 × f32>), res_attrs = [{}, {test.erase_this_result}, {}, {test.erase_this_result}, {}], sym_name = "f", sym_visibility = "private"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZtest-func-erase-result.out.txt" str
