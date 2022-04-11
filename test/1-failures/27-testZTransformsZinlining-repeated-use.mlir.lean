
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
    "func.call"() {callee = @B, inA} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "A"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @E} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "B"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @D} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "C"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @B, inD} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "D", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "func.call"() {callee = @fabsf} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "E"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "fabsf", sym_visibility = "private"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZinlining-repeated-use.out.txt" str
