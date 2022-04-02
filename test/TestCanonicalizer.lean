
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
  ^bb0(%arg0: memref<4 × f32>, %arg1: memref<? × f32>, %arg2: memref<64 × 16 × 4 × f32, #map2>):
    %0 = "memref.cast"(%arg0) : (memref<4 × f32>) -> memref<? × f32>
    %1 = "memref.cast"(%arg1) : (memref<? × f32>) -> memref<4 × f32>
    %2 = "memref.cast"(%arg2) : (memref<64 × 16 × 4 × f32, #map2>) -> memref<64 × 16 × 4 × f32, #map3>
    %3 = "memref.cast"(%2) : (memref<64 × 16 × 4 × f32, #map3>) -> memref<64 × 16 × 4 × f32, #map2>
    %5 = "memref.cast"(%4) : (memref<* × f32>) -> memref<4 × f32>
    "func.return"() : () -> ()
  }) {function_type = (memref<4 × f32>, memref<? × f32>, memref<64 × 16 × 4 × f32, #map2>) -> (), sym_name = "memref_cast"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZcore-ops.out.txt" str
