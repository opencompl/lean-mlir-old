
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
  "test.top_level_op"() : () -> ()
  "test.top_level_op"() : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.top_level_op_strict_loc"() {strict_loc_check} : () -> ()
  "test.top_level_op_strict_loc"() {strict_loc_check} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.top_level_op_loc_match"() {strict_loc_check} : () -> ()
  "test.top_level_op_loc_match"() {strict_loc_check} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.top_level_op_block_loc_mismatch"() ({
  ^bb0(%arg0: i32):
  }) {strict_loc_check} : () -> ()
  "test.top_level_op_block_loc_mismatch"() ({
  ^bb0(%arg0: i32):
  }) {strict_loc_check} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.top_level_op_block_loc_match"() ({
  ^bb0(%arg0: i32):
  }) {strict_loc_check} : () -> ()
  "test.top_level_op_block_loc_match"() ({
  ^bb0(%arg0: i32):
  }) {strict_loc_check} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.top_level_name_mismatch"() : () -> ()
  "test.top_level_name_mismatch2"() : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.top_level_op_attr_mismatch"() {foo = "bar"} : () -> ()
  "test.top_level_op_attr_mismatch"() {foo = "bar2"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.top_level_op_cfg"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0)[^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%0: f32):  
    "test.some_branching_op"() : () -> ()
  ^bb2(%1: i32):  
    "test.some_branching_op"() : () -> ()
  }, {
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0)[^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%0: f32):  
    "test.some_branching_op"() : () -> ()
  ^bb2(%1: i32):  
    "test.some_branching_op"() : () -> ()
  }) {attr = "foo"} : () -> ()
  "test.top_level_op_cfg"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0)[^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%0: f32):  
    "test.some_branching_op"() : () -> ()
  ^bb2(%1: i32):  
    "test.some_branching_op"() : () -> ()
  }, {
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0)[^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%0: f32):  
    "test.some_branching_op"() : () -> ()
  ^bb2(%1: i32):  
    "test.some_branching_op"() : () -> ()
  }) {attr = "foo"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.operand_num_mismatch"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0) : (f32, i32) -> ()
  }) : () -> ()
  "test.operand_num_mismatch"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1) : (f32) -> ()
  }) : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.operand_type_mismatch"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg0) : (f32, i32) -> ()
  }) : () -> ()
  "test.operand_type_mismatch"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"(%arg1, %arg1) : (f32, f32) -> ()
  }) : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.block_type_mismatch"() ({
  ^bb0(%arg0: f32, %arg1: f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()
  "test.block_type_mismatch"() ({
  ^bb0(%arg0: i32, %arg1: f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.block_arg_num_mismatch"() ({
  ^bb0(%arg0: f32, %arg1: f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()
  "test.block_arg_num_mismatch"() ({
  ^bb0(%arg0: f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.dataflow_match"() ({
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()
  "test.dataflow_match"() ({
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()
}) : () -> ()


"builtin.module"() ({
  "test.dataflow_mismatch"() ({
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()
  "test.dataflow_mismatch"() ({
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.consumer"(%0#1, %0#0) : (i32, i32) -> ()
  }) : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZoperation-equality.out.txt" str
