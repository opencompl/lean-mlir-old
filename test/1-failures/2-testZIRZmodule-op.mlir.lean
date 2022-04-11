
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
^bb0:
}) : () -> ()


"builtin.module"() ({
^bb0:
}) {foo.attr = true} : () -> ()


"builtin.module"() ({
  %0 = "foo.result_op"() : () -> i32
}) : () -> ()


"builtin.module"() ({
^bb0:
}) : () -> ()


"builtin.module"() ({
^bb0:
}) : () -> ()


"builtin.module"() ({
  %0 = "op"() : () -> i32
}) : () -> ()


"builtin.module"() ({
  "builtin.module"() ({
    "builtin.module"() ({
    ^bb0:
    }) {foo.bar, sym_name = "bar"} : () -> ()
  }) : () -> ()
}) {sym_name = "foo"} : () -> ()


"builtin.module"() ({
^bb0:
}) {test.another_attribute = #dlti.dl_spec<>, test.random_attribute = #dlti.dl_spec<>} : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZmodule-op.out.txt" str
