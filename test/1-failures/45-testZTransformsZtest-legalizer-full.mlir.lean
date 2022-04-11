
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
    %0 = "test.type_producer"() : () -> i32
    "test.type_consumer"(%0) : (i32) -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "multi_level_mapping"} : () -> ()
  "func.func"() ({
    "test.drop_region_op"() ({
      %0 = "test.illegal_op_f"() : () -> i32
      "test.return"() : () -> ()
    }) : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "dropped_region_with_illegal_ops"} : () -> ()
  "func.func"() ({
    %0 = "test.replace_non_root"() : () -> i32
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "replace_non_root_illegal_op"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "builtin.module"() ({
      %0 = "test.illegal_op_f"() : () -> i32
    }) {test.recursively_legal} : () -> ()
    "func.func"() ({
    ^bb0(%arg0: i64):
      %0 = "test.illegal_op_f"() : () -> i32
      "test.return"() : () -> ()
    }) {function_type = (i64) -> (), sym_name = "dynamic_func", test.recursively_legal} : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "recursively_legal_invalid_op"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.region"() ({
    ^bb0(%arg0: i64):
      "test.invalid"(%arg0) : (i64) -> ()
    }) {legalizer.should_clone} : () -> ()
    %0 = "test.illegal_op_f"() : () -> i32
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_undo_region_clone"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "foo.unknown_op"() {test.dynamically_legal} : () -> ()
    "foo.unknown_op"() : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_unknown_dynamically_legal"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.region"() ({
    ^bb0(%arg0: i64):
      "cf.br"(%arg0)[^bb1] : (i64) -> ()
    ^bb1(%0: i64):  
      "test.invalid"(%0) : (i64) -> ()
    }) : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_undo_region_inline"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.region"() ({
    ^bb0(%arg0: i64):
      "cf.br"(%arg0)[^bb1] : (i64) -> ()
    ^bb1(%0: i64):  
      "test.invalid"(%0) : (i64) -> ()
    }) {legalizer.erase_old_blocks, legalizer.should_clone} : () -> ()
    "test.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_undo_block_erase"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.illegal_op_g"() : () -> i32
    "test.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "create_unregistered_op_in_pattern"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZtest-legalizer-full.out.txt" str
