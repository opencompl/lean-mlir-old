
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
  quoteMList xs.toList

  
-- | write an op into the path
def o: List Op := [mlir_ops|





















































"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, 1>) -> (), sym_name = "f0", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, 1>) -> (), sym_name = "f1", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<i8, #map0, 1>) -> (), sym_name = "f2", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3a", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3b", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3c", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3d", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3e", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3f", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3g", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3h", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3i", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3j", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3k", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map2, 1>) -> (), sym_name = "f3l", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map3, 1>) -> (), sym_name = "f4", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map4, 1>) -> (), sym_name = "f5", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map5, 1>) -> (), sym_name = "f6", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map6, 1>) -> (), sym_name = "f7", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map7, 1>) -> (), sym_name = "f8", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map8, 1>) -> (), sym_name = "f9", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map9, 1>) -> (), sym_name = "f10", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map10, 1>) -> (), sym_name = "f11", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map11, 1>) -> (), sym_name = "f12", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map12, 1>) -> (), sym_name = "f13", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map13, 1>) -> (), sym_name = "f14", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map14, 1>) -> (), sym_name = "f15", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map15, 1>) -> (), sym_name = "f16", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map16, 1>) -> (), sym_name = "f17", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map17, 1>) -> (), sym_name = "f19", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map18, 1>) -> (), sym_name = "f20", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map19, 1>) -> (), sym_name = "f18", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map20, 1>) -> (), sym_name = "f21", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map21, 1>) -> (), sym_name = "f22", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map22, 1>) -> (), sym_name = "f23", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map23, 1>) -> (), sym_name = "f24", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map24, 1>) -> (), sym_name = "f25", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map25, 1>) -> (), sym_name = "f26", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map26, 1>) -> (), sym_name = "f29", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map27, 1>) -> (), sym_name = "f30", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map28, 1>) -> (), sym_name = "f32", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map29, 1>) -> (), sym_name = "f33", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map30, 1>) -> (), sym_name = "f34", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × 4 × i8, #map31, 1>) -> (), sym_name = "f35", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map32, 1>) -> (), sym_name = "f36", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map33, 1>) -> (), sym_name = "f37", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map34, 1>) -> (), sym_name = "f38", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map35, 1>) -> (), sym_name = "f39", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map36>) -> (), sym_name = "f43", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map37>) -> (), sym_name = "f44", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × 100 × i8, #map38>) -> (), sym_name = "f45", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × 100 × i8, #map39>) -> (), sym_name = "f46", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × 100 × i8, #map40>) -> (), sym_name = "f47", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × 100 × i8, #map41>) -> (), sym_name = "f48", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × i8, #map42>) -> (), sym_name = "f49", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × i8, #map43>) -> (), sym_name = "f50", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × i8, #map44>) -> (), sym_name = "f51", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × i8, #map45>) -> (), sym_name = "f52", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × i8, #map46>) -> (), sym_name = "f53", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<10 × i32, #map47>) -> (), sym_name = "f54", sym_visibility = "private"} : () -> ()
  "foo.op"() {map = #map48} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × 1 × i8, #map49>) -> (), sym_name = "f56", sym_visibility = "private"} : () -> ()
  "f57"() {map = #map50} : () -> ()
  "f58"() {map = #map51} : () -> ()
  "f59"() {map = #map52} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZaffine-map.out.txt" str
