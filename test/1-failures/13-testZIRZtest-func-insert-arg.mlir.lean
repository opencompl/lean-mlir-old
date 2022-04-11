
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
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "f", test.insert_args = [[0, i1, {test.A}]]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i2):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.B}], function_type = (i2) -> (), sym_name = "f", test.insert_args = [[0, i1, {test.A}]]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}], function_type = (i1) -> (), sym_name = "f", test.insert_args = [[1, i2, {test.B}]]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i1, %arg1: i3):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}, {test.C}], function_type = (i1, i3) -> (), sym_name = "f", test.insert_args = [[1, i2, {test.B}]]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i2):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.B}], function_type = (i2) -> (), sym_name = "f", test.insert_args = [[0, i1, {test.A}], [1, i3, {test.C}]]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i3):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.C}], function_type = (i3) -> (), sym_name = "f", test.insert_args = [[0, i1, {test.A}], [0, i2, {test.B}]]} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZtest-func-insert-arg.out.txt" str
