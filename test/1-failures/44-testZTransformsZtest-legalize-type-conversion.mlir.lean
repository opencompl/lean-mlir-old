
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
  ^bb0(%arg0: i16):
    "foo.return"(%arg0) : (i16) -> ()
  }) {function_type = (i16) -> (), sym_name = "test_invalid_arg_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i64):
    "foo.return"(%arg0) : (i64) -> ()
  }) {function_type = (i64) -> (), sym_name = "test_valid_arg_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.type_producer"() : () -> f16
    "foo.return"(%0) : (f16) -> ()
  }) {function_type = () -> (), sym_name = "test_invalid_result_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.type_producer"() : () -> f16
    "foo.return"(%0) : (f16) -> ()
  }) {function_type = () -> (), sym_name = "test_invalid_result_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.another_type_producer"() : () -> f32
    "foo.return"(%0) : (f32) -> ()
  }) {function_type = () -> (), sym_name = "test_transitive_use_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.another_type_producer"() : () -> f16
    "foo.return"(%0) : (f16) -> ()
  }) {function_type = () -> (), sym_name = "test_transitive_use_invalid_materialization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.type_producer"() : () -> f32
    "foo.return"(%0) : (f32) -> ()
  }) {function_type = () -> (), sym_name = "test_valid_result_legalization"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.signature_conversion_undo"() ({
    ^bb0(%arg0: f32):
      "test.type_consumer"(%arg0) : (f32) -> ()
      "test.return"(%arg0) : (f32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_signature_conversion_undo"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.unsupported_block_arg_type"() ({
    ^bb0(%arg0: index):
      "test.return"(%arg0) : (index) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_block_argument_not_converted"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "test.signature_conversion_no_converter"() ({
    ^bb0(%arg0: f32):
      "test.type_consumer"(%arg0) : (f32) -> ()
      "test.return"(%arg0) : (f32) -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "test_signature_conversion_no_converter"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.type_producer"() : () -> !test.test_rec<something, test_rec<something>>
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "recursive_type_conversion"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZtest-legalize-type-conversion.out.txt" str
