
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
    %0 = "test.source"() : () -> memref<f32>
    "func.return"(%0) : (memref<f32>) -> ()
  }) {function_type = () -> memref<f32>, sym_name = "basic"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<1 × f32>):
    %0 = "test.source"() : () -> memref<2 × f32>
    "func.return"(%0) : (memref<2 × f32>) -> ()
  }) {function_type = (memref<1 × f32>) -> memref<2 × f32>, sym_name = "presence_of_existing_arguments"} : () -> ()
  "func.func"() ({
    %0:2 = "test.source"() : () -> (memref<1 × f32>, memref<2 × f32>)
    "func.return"(%0#0, %0#1) : (memref<1 × f32>, memref<2 × f32>) -> ()
  }) {function_type = () -> (memref<1 × f32>, memref<2 × f32>), sym_name = "multiple_results"} : () -> ()
  "func.func"() ({
    %0:3 = "test.source"() : () -> (i1, memref<f32>, i32)
    "func.return"(%0#0, %0#1, %0#2) : (i1, memref<f32>, i32) -> ()
  }) {function_type = () -> (i1, memref<f32>, i32), sym_name = "non_memref_types"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> memref<f32>, sym_name = "external_function", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> memref<f32>, res_attrs = [{test.some_attr}], sym_name = "result_attrs", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (memref<1 × f32>, memref<2 × f32>, memref<3 × f32>), res_attrs = [{}, {test.some_attr}, {}], sym_name = "mixed_result_attrs", sym_visibility = "private"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> memref<1 × f32>, sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "func.call"() {callee = @callee} : () -> memref<1 × f32>
    "test.sink"(%0) : (memref<1 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "call_basic"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (memref<1 × f32>, memref<2 × f32>), sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0:2 = "func.call"() {callee = @callee} : () -> (memref<1 × f32>, memref<2 × f32>)
    "test.sink"(%0#0, %0#1) : (memref<1 × f32>, memref<2 × f32>) -> ()
  }) {function_type = () -> (), sym_name = "call_multiple_result"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (i1, memref<1 × f32>, i32), sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0:3 = "func.call"() {callee = @callee} : () -> (i1, memref<1 × f32>, i32)
    "test.sink"(%0#0, %0#1, %0#2) : (i1, memref<1 × f32>, i32) -> ()
  }) {function_type = () -> (), sym_name = "call_non_memref_result"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> memref<? × f32>, sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "func.call"() {callee = @callee} : () -> memref<? × f32>
    "test.sink"(%0) : (memref<? × f32>) -> ()
  }) {function_type = () -> (), sym_name = "call_non_memref_result"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZbuffer-results-to-out-params.out.txt" str
