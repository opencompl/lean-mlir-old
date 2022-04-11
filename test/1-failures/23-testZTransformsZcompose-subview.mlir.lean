
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
  ^bb0(%arg0: memref<4 × 1024 × f32>):
    %0 = "memref.subview"(%arg0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [2, 256], static_sizes = [2, 256], static_strides = [1, 1]} : (memref<4 × 1024 × f32>) -> memref<2 × 256 × f32, #map0>
    %1 = "memref.subview"(%0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [1, 128], static_sizes = [1, 128], static_strides = [1, 1]} : (memref<2 × 256 × f32, #map0>) -> memref<1 × 128 × f32, #map1>
    "func.return"(%1) : (memref<1 × 128 × f32, #map1>) -> ()
  }) {function_type = (memref<4 × 1024 × f32>) -> memref<1 × 128 × f32, #map1>, sym_name = "main"} : () -> ()
}) : () -> ()





"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<4 × 1024 × f32>):
    %0 = "memref.subview"(%arg0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [1, 512], static_sizes = [3, 256], static_strides = [1, 1]} : (memref<4 × 1024 × f32>) -> memref<3 × 256 × f32, #map0>
    %1 = "memref.subview"(%0) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [1, 128], static_sizes = [2, 128], static_strides = [1, 1]} : (memref<3 × 256 × f32, #map0>) -> memref<2 × 128 × f32, #map1>
    %2 = "memref.subview"(%1) {operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4 × i32>, static_offsets = [1, 33], static_sizes = [1, 10], static_strides = [1, 1]} : (memref<2 × 128 × f32, #map1>) -> memref<1 × 10 × f32, #map2>
    "func.return"(%2) : (memref<1 × 10 × f32, #map2>) -> ()
  }) {function_type = (memref<4 × 1024 × f32>) -> memref<1 × 10 × f32, #map2>, sym_name = "main"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<4 × 1024 × f32>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "arith.constant"() {value = 2 : index} : () -> index
    %2 = "memref.subview"(%arg0, %1) {operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4 × i32>, static_offsets = [-9223372036854775808, 256], static_sizes = [2, 256], static_strides = [1, 1]} : (memref<4 × 1024 × f32>, index) -> memref<2 × 256 × f32, #map>
    %3 = "memref.subview"(%2, %0) {operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4 × i32>, static_offsets = [-9223372036854775808, 128], static_sizes = [1, 128], static_strides = [1, 1]} : (memref<2 × 256 × f32, #map>, index) -> memref<1 × 128 × f32, #map>
    "func.return"(%3) : (memref<1 × 128 × f32, #map>) -> ()
  }) {function_type = (memref<4 × 1024 × f32>) -> memref<1 × 128 × f32, #map>, sym_name = "main"} : () -> ()
}) : () -> ()



"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<4 × 1024 × f32>):
    %0 = "arith.constant"() {value = 2 : index} : () -> index
    %1 = "arith.constant"() {value = 128 : index} : () -> index
    %2 = "memref.subview"(%arg0, %0) {operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4 × i32>, static_offsets = [-9223372036854775808, 256], static_sizes = [2, 256], static_strides = [1, 1]} : (memref<4 × 1024 × f32>, index) -> memref<2 × 256 × f32, #map>
    %3 = "memref.subview"(%2, %1) {operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4 × i32>, static_offsets = [1, -9223372036854775808], static_sizes = [1, 128], static_strides = [1, 1]} : (memref<2 × 256 × f32, #map>, index) -> memref<1 × 128 × f32, #map>
    "func.return"(%3) : (memref<1 × 128 × f32, #map>) -> ()
  }) {function_type = (memref<4 × 1024 × f32>) -> memref<1 × 128 × f32, #map>, sym_name = "main"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZcompose-subview.out.txt" str
