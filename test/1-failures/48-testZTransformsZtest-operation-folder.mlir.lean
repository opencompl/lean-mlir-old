
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
    %0 = "arith.constant"() {value = 42 : i32} : () -> i32
    %1 = "test.op_in_place_fold_anchor"(%0) : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "foo"} : () -> ()
  "func.func"() ({
    %0 = "test.cast"() {test_fold_before_previously_folded_op} : () -> i32
    %1 = "test.cast"() {test_fold_before_previously_folded_op} : () -> i32
    "func.return"(%0, %1) : (i32, i32) -> ()
  }) {function_type = () -> (i32, i32), sym_name = "test_fold_before_previously_folded_op"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZtest-operation-folder.out.txt" str
