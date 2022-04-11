
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
  "test.dense_attr"() {foo.dense_attr = dense<[1, 2, 3]> : tensor<3 × i32>} : () -> ()
  "test.non_elided_dense_attr"() {foo.dense_attr = dense<[1, 2]> : tensor<2 × i32>} : () -> ()
  "test.sparse_attr"() {foo.sparse_attr = sparse<[[0, 0, 5]], -2.00000000> : vector<1 × 1 × 10 × f16>} : () -> ()
  "test.opaque_attr"() {foo.opaque_attr = opaque<"elided_large_const", "0 × EBFE"> : tensor<100 × f32>} : () -> ()
  "test.dense_splat"() {foo.dense_attr = dense<1> : tensor<3 × i32>} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZpretty-attributes.out.txt" str
