
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






!test_tuple = type tuple<!test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla>
!test_ui8_ = type !test.int<unsigned, 8>
!tuple = type tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>
"builtin.module"() ({
  "test.op"() {alias_test = #test2Ealias} : () -> ()
  "test.op"() {alias_test = #test_alias0_} : () -> ()
  "test.op"() {alias_test = #_0_test_alias} : () -> ()
  "test.op"() {alias_test = [#test_alias_conflict0_0, #test_alias_conflict0_1]} : () -> ()
  %0 = "test.op"() {alias_test = "alias_test:large_tuple"} : () -> !tuple
  %1 = "test.op"() {alias_test = "alias_test:large_tuple"} : () -> !test_tuple
  %2 = "test.op"() : () -> tensor<32 × f32, #test_encoding>
  %3 = "test.op"() : () -> tensor<32 × !test_ui8_>
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZprint-attr-type-aliases.out.txt" str
