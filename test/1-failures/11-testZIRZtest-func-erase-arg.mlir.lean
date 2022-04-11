
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
  ^bb0(%arg0: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.erase_this_arg}], function_type = (f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.erase_this_arg}, {test.A}], function_type = (f32, f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}, {test.erase_this_arg}], function_type = (f32, f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}, {test.erase_this_arg}, {test.B}], function_type = (f32, f32, f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}, {test.erase_this_arg}, {test.erase_this_arg}, {test.B}], function_type = (f32, f32, f32, f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32):
    "func.return"() : () -> ()
  }) {arg_attrs = [{test.A}, {test.erase_this_arg}, {test.B}, {test.erase_this_arg}, {test.C}], function_type = (f32, f32, f32, f32, f32) -> (), sym_name = "f"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<1 × f32>, %arg1: f32, %arg2: tensor<2 × f32>, %arg3: f32, %arg4: tensor<3 × f32>):
    "func.return"() : () -> ()
  }) {arg_attrs = [{}, {test.erase_this_arg}, {}, {test.erase_this_arg}, {}], function_type = (tensor<1 × f32>, f32, tensor<2 × f32>, f32, tensor<3 × f32>) -> (), sym_name = "f"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZtest-func-erase-arg.out.txt" str
