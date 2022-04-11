
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
  }) {function_type = () -> (), sym_name = "symbol_foo", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "foo.op"() {non_symbol_attr, use = [{nested_symbol = [@symbol_foo]}], z_other_non_symbol_attr} : () -> ()
  }) {function_type = () -> (), sym.use = @symbol_foo, sym_name = "symbol_bar"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "symbol_removable", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "symbol_baz", sym_visibility = "private"} : () -> ()
  "builtin.module"() ({
    "foo.op"() {test.nested_reference = @symbol_baz} : () -> ()
  }) {test.reference = @symbol_baz} : () -> ()
}) {sym.outside_use = @symbol_foo} : () -> ()


"builtin.module"() ({
  "builtin.module"() ({
    "builtin.module"() ({
      "func.func"() ({
      }) {function_type = () -> (), sym_name = "foo", sym_visibility = "nested"} : () -> ()
    }) {sym_name = "module_c"} : () -> ()
  }) {sym_name = "module_b"} : () -> ()
  "func.func"() ({
    "foo.op"() {use_1 = [{nested_symbol = [@module_b::@module_c::@foo]}], use_2 = @module_b} : () -> ()
  }) {function_type = () -> (), sym_name = "symbol_bar"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "foo.possibly_unknown_symbol_table"() ({
    }) : () -> ()
  }) {function_type = () -> (), sym_name = "symbol_bar"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZtest-symbol-uses.out.txt" str
