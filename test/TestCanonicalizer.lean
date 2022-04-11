
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
  ^bb0(%arg0: i1, %arg1: i32):
    %0 = "arith.constant"() {value = true} : () -> i1
    %1 = "arith.constant"() {value = -10 : i32} : () -> i32
    %2 = "arith.constant"() {value = 31 : i32} : () -> i32
    %3 = "arith.andi"(%arg0, %0) : (i1, i1) -> i1
    %4 = "arith.andi"(%arg1, %1) : (i32, i32) -> i32
    %5 = "arith.andi"(%4, %2) : (i32, i32) -> i32
    "func.return"(%3, %5) : (i1, i32) -> ()
  }) {function_type = (i1, i32) -> (i1, i32), sym_name = "simple_and"} : () -> ()
}) : () -> ()

] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZconstant-fold.out.txt" str
