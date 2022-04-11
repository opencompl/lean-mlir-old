
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
  ^bb0(%arg0: i32, %arg1: i32):
    %0:2 = "test.merge_blocks"() ({
      "test.br"(%arg0, %arg1)[^bb1] : (i32, i32) -> ()
    ^bb1(%1: i32, %2: i32):  
      "test.return"(%1, %2) : (i32, i32) -> ()
    }) : () -> (i32, i32)
    "test.return"(%0#0, %0#1) : (i32, i32) -> ()
  }) {function_type = (i32, i32) -> (), sym_name = "merge_blocks"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    "test.undo_blocks_merge"() ({
      "unregistered.return"(%arg0)[^bb1] : (i32) -> ()
    ^bb1(%0: i32):  
      "unregistered.return"(%0) : (i32) -> ()
    }) : () -> ()
  }) {function_type = (i32) -> (), sym_name = "undo_blocks_merge"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.SingleBlockImplicitTerminator"() ({
      %0 = "test.type_producer"() : () -> i32
      "test.SingleBlockImplicitTerminator"() ({
        "test.type_consumer"(%0) : (i32) -> ()
        "test.finish"() : () -> ()
      }) : () -> ()
      "test.finish"() : () -> ()
    }) : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "inline_regions"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZtest-merge-blocks.out.txt" str
