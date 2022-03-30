
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
  quoteMList xs.toList

  
-- | write an op into the path
def o: List Op := [mlir_ops|
"builtin.module"() ({
  "shape.function_library"() ({
    "func.func"() ({
    ^bb0(%arg0: i32):
      "func.return"(%0) : (i32) -> ()
    }) {function_type = (i32) -> i32, sym_name = "same_result_shape"} : () -> ()
  }) {mapping = {test.same_operand_result_type = @same_result_shape}, sym_name = "shape_lib"} : () -> ()
}) {shape.lib = [@shape_lib]} : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZAnalysisZtest-shape-fn-report.out.txt" str
