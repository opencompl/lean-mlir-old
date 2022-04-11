
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
  ^bb0(%arg0: f32, %arg1: f32):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 10 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    %3 = "scf.for"(%0, %1, %2, %arg0) ({
    ^bb0(%arg2: index, %arg3: f32):
      %4 = "arith.addf"(%arg3, %arg1) : (f32, f32) -> f32
      "scf.yield"(%4) : (f32) -> ()
    }) : (index, index, index, f32) -> f32
    "func.return"(%3) : (f32) -> ()
  }) {function_type = (f32, f32) -> f32, sym_name = "scf_loop_unroll_single"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: index):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2:2 = "scf.for"(%0, %arg2, %1, %arg0, %arg1) ({
    ^bb0(%arg3: index, %arg4: f32, %arg5: f32):
      %3 = "arith.addf"(%arg4, %arg0) : (f32, f32) -> f32
      %4 = "arith.addf"(%arg5, %arg1) : (f32, f32) -> f32
      "scf.yield"(%3, %4) : (f32, f32) -> ()
    }) : (index, index, index, f32, f32) -> (f32, f32)
    "func.return"(%2#0, %2#1) : (f32, f32) -> ()
  }) {function_type = (f32, f32, index) -> (f32, f32), sym_name = "scf_loop_unroll_double_symbolic_ub"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    "scf.for"(%1, %2, %0) ({
    ^bb0(%arg0: index):
      %3 = "test.foo"(%arg0) : (index) -> i32
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "scf_loop_unroll_factor_1_promote"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZscf-loop-unroll.out.txt" str
