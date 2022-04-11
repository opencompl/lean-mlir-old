
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
  %0 = "test.side_effect_op"() : () -> i32
  %1 = "test.side_effect_op"() {effects = [{effect = "read"}, {effect = "free"}]} : () -> i32
  %2 = "test.side_effect_op"() {effects = [{effect = "write", test_resource}]} : () -> i32
  %3 = "test.side_effect_op"() {effects = [{effect = "allocate", on_result, test_resource}]} : () -> i32
  %4 = "test.side_effect_op"() {effects = [{effect = "read", on_reference = @foo_ref, test_resource}]} : () -> i32
  %5 = "test.side_effect_op"() {effect_parameter = #map} : () -> i32
  %6 = "test.unregistered_side_effect_op"() {effect_parameter = #map} : () -> i32
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZtest-side-effects.out.txt" str
