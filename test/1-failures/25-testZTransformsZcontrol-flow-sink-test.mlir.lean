
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
    %0 = "test.sink_me"() : () -> i32
    "test.sink_target"() ({
      "test.use"(%0) : (i32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_sink"} : () -> ()
  "func.func"() ({
    %0 = "test.sink_me"() {first} : () -> i32
    %1 = "test.sink_me"() {second} : () -> i32
    "test.sink_target"() ({
      "test.use"(%0) : (i32) -> ()
    }, {
      "test.use"(%1) : (i32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_sink_first_region_only"} : () -> ()
  "func.func"() ({
    %0 = "test.sink_me"() : () -> i32
    %1 = "test.dont_sink_me"() : () -> i32
    "test.sink_target"() ({
      "test.use"(%0, %1) : (i32, i32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_sink_targeted_op_only"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZcontrol-flow-sink-test.out.txt" str
