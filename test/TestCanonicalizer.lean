
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
  "func.func"() ({
  ^bb0(%arg0: tensor<10 × 20 × f32>):
    %0 = "math.tanh"(%arg0) : (tensor<10 × 20 × f32>) -> tensor<10 × 20 × f32>
    %1 = "test.same_operand_result_type"(%0) : (tensor<10 × 20 × f32>) -> tensor<10 × 20 × f32>
    "func.return"(%1) : (tensor<10 × 20 × f32>) -> ()
  }) {function_type = (tensor<10 × 20 × f32>) -> tensor<10 × 20 × f32>, shape.function = @shape_lib::@same_result_shape, sym_name = "tanh"} : () -> ()
  "shape.function_library"() ({
    "func.func"() ({
    ^bb0(%arg0: !shape.value_shape):
      %0 = "shape.shape_of"(%arg0) : (!shape.value_shape) -> !shape.shape
      "func.return"(%0) : (!shape.shape) -> ()
    }) {function_type = (!shape.value_shape) -> !shape.shape, sym_name = "same_result_shape"} : () -> ()
  }) {mapping = {test.same_operand_result_type = @same_result_shape}, sym_name = "shape_lib"} : () -> ()
}) {shape.lib = [@shape_lib]} : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZAnalysisZtest-shape-fn-report.out.txt" str
