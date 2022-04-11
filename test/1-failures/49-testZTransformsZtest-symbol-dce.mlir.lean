
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
  }) {function_type = () -> (), sym_name = "dead_private_function", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "dead_nested_function", sym_visibility = "nested"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "live_private_function", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "live_nested_function", sym_visibility = "nested"} : () -> ()
  "func.func"() ({
    "foo.return"() {uses = [@live_private_function, @live_nested_function]} : () -> ()
  }) {function_type = () -> (), sym_name = "public_function"} : () -> ()
}) {test.simple} : () -> ()


"builtin.module"() ({
  "builtin.module"() ({
    "func.func"() ({
    }) {function_type = () -> (), sym_name = "dead_nested_function", sym_visibility = "nested"} : () -> ()
    "func.func"() ({
    }) {function_type = () -> (), sym_name = "private_function", sym_visibility = "private"} : () -> ()
    "func.func"() ({
      "foo.return"() {uses = [@private_function]} : () -> ()
    }) {function_type = () -> (), sym_name = "nested_function", sym_visibility = "nested"} : () -> ()
  }) {sym_name = "public_module"} : () -> ()
  "live.user"() {uses = [@public_module::@nested_function]} : () -> ()
}) {test.nested} : () -> ()


"builtin.module"() ({
  "builtin.module"() ({
    "func.func"() ({
    }) {function_type = () -> (), sym_name = "nested_function", sym_visibility = "nested"} : () -> ()
  }) {sym_name = "public_module"} : () -> ()
  "builtin.module"() ({
    "func.func"() ({
    }) {function_type = () -> (), sym_name = "nested_function", sym_visibility = "nested"} : () -> ()
  }) {sym_name = "nested_module", sym_visibility = "nested"} : () -> ()
  "builtin.module"() ({
    "func.func"() ({
    }) {function_type = () -> (), sym_name = "nested_function", sym_visibility = "nested"} : () -> ()
  }) {sym_name = "private_module", sym_visibility = "private"} : () -> ()
  "live.user"() {uses = [@nested_module, @private_module]} : () -> ()
}) {test.no_dce_non_hidden_parent} : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "private_symbol", sym_visibility = "private"} : () -> ()
  "foo.possibly_unknown_symbol_table"() ({
  }) : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "dead_private_function", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "live_private_function", sym_visibility = "private"} : () -> ()
  "live.user"() {uses = [@live_private_function]} : () -> ()
  "live.user"() {uses = [@unknown_symbol]} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZtest-symbol-dce.out.txt" str
