
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
  }) {foo = #test.cmpnd_a<1, !test.smpla, [5, 6]>, function_type = () -> (), sym_name = "compoundA", sym_visibility = "private"} : () -> ()
  %0 = "test.result_has_same_type_as_attr"() {attr = #test<"attr_with_self_type_param i32">} : () -> i32
  %1 = "test.result_has_same_type_as_attr"() {attr = #test<"attr_with_type_builder 10 : i16">} : () -> i16
  "func.func"() ({
  }) {foo = #test.cmpnd_nested_outer_qual<i #test.cmpnd_nested_inner<42 <1, !test.smpla, [5, 6]>>>, function_type = () -> (), sym_name = "qualifiedAttr", sym_visibility = "private"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZmlir-tblgenZtestdialect-attrdefs.out.txt" str
