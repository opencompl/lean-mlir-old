
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
  }) {attr0 = #test.attr_with_format<3 : two = "hello", four = [1, 2, 3] : 42 : i64>, attr1 = #test.attr_with_format<5 : two = "a_string", four = [4, 5, 6, 7, 8] : 8 : i8>, attr2 = #test<"attr_ugly begin 5 : inde ×  end">, attr3 = #test.attr_params<42, 24>, attr4 = #test.attr_with_type<i32, vector<4 × i32>>, function_type = (!test.type_with_format<111, three = #test<"attr_ugly begin 5 : inde ×  end">, two = "foo">) -> !test.type_with_format<2147, three = "hi", two = "hi">, sym_name = "test_roundtrip_parameter_parsers", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (!test.no_parser<255, [1, 2, 3, 4, 5], "foobar", 4>) -> (!test.struct_capture_all<v0 = 0, v1 = 1, v2 = 2, v3 = 3>, !test.optional_param<, 6>, !test.optional_param<5, 6>, !test.optional_params<"a">, !test.optional_params<5, "a">, !test.optional_struct<b = "a">, !test.optional_struct<a = 5, b = "a">, !test.optional_params_after<"a">, !test.optional_params_after<"a", 5>, !test.all_optional_params<>, !test.all_optional_params<5>, !test.all_optional_params<5, 6>, !test.all_optional_struct<>, !test.all_optional_struct<b = 5>, !test.all_optional_struct<a = 5, b = 10>, !test.optional_group<(5) 6>, !test.optional_group< ×  6>, !test.optional_group_params< × >, !test.optional_group_params<(5)>, !test.optional_group_params<(5, 6)>, !test.optional_group_struct< × >, !test.optional_group_struct<(b = 5)>, !test.optional_group_struct<(a = 10, b = 5)>, !test.spaces< 5
()() 6>, !test.ap_float<5.00000000>, !test.ap_float<>, !test.default_valued_type<(i64)>, !test.default_valued_type<>, !test.custom_type<-5>, !test.custom_type<2 0 1 5>, !test.custom_type_string<"foo" foo>, !test.custom_type_string<"bar" bar>), sym_name = "test_roundtrip_default_parsers_struct", sym_visibility = "private"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZmlir-tblgenZattr-or-type-format-roundtrip.out.txt" str
