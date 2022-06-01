
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
  quoteMList xs.toList (<- `(MLIR.AST.Op))

  
-- | write an op into the path
def o: List Op := [mlir_ops|
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: tensor<? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "tensor.dim"(%arg0, %0) : (tensor<? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, index) -> index
    "func.return"(%1) : (index) -> ()
  }) {function_type = (tensor<? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> index, sym_name = "sparse_dim1d"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<? × ? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "tensor.dim"(%arg0, %0) : (tensor<? × ? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>, index) -> index
    "func.return"(%1) : (index) -> ()
  }) {function_type = (tensor<? × ? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>) -> index, sym_name = "sparse_dim3d"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<10 × 20 × 30 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>):
    %0 = "arith.constant"() {value = 1 : index} : () -> index
    %1 = "tensor.dim"(%arg0, %0) : (tensor<10 × 20 × 30 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>, index) -> index
    "func.return"(%1) : (index) -> ()
  }) {function_type = (tensor<10 × 20 × 30 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>) -> index, sym_name = "sparse_dim3d_const"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !llvm<"ptr<i8>">):
    %0 = "sparse_tensor.new"(%arg0) : (!llvm<"ptr<i8>">) -> tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (!llvm<"ptr<i8>">) -> tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_new1d"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !llvm<"ptr<i8>">):
    %0 = "sparse_tensor.new"(%arg0) : (!llvm<"ptr<i8>">) -> tensor<? × ? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<? × ? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (!llvm<"ptr<i8>">) -> tensor<? × ? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_new2d"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !llvm<"ptr<i8>">):
    %0 = "sparse_tensor.new"(%arg0) : (!llvm<"ptr<i8>">) -> tensor<? × ? × ? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>
    "func.return"(%0) : (tensor<? × ? × ? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>) -> ()
  }) {function_type = (!llvm<"ptr<i8>">) -> tensor<? × ? × ? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>, sym_name = "sparse_new3d"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index):
    %0 = "sparse_tensor.init"(%arg0, %arg1) : (index, index) -> tensor<? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (index, index) -> tensor<? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_init"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    "sparse_tensor.release"(%arg0) : (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
    "func.return"() : () -> ()
  }) {function_type = (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> (), sym_name = "sparse_release"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<64 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "sparse_tensor.convert"(%arg0) : (tensor<64 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> tensor<64 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<64 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (tensor<64 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> tensor<64 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_nop_convert"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<32 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "sparse_tensor.convert"(%arg0) : (tensor<32 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (tensor<32 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_hidden_nop_cast"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<64 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "tensor.cast"(%arg0) : (tensor<64 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (tensor<64 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_nop_cast"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<? × i32>):
    %0 = "sparse_tensor.convert"(%arg0) : (tensor<? × i32>) -> tensor<? × i32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<? × i32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (tensor<? × i32>) -> tensor<? × i32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_convert_1d"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<100 × comple × <f64>>):
    %0 = "sparse_tensor.convert"(%arg0) : (tensor<100 × comple × <f64>>) -> tensor<100 × comple × <f64>, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<100 × comple × <f64>, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (tensor<100 × comple × <f64>>) -> tensor<100 × comple × <f64>, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_convert_complex"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 64, inde × BitWidth = 64 }>">>):
    %0 = "sparse_tensor.convert"(%arg0) : (tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 64, inde × BitWidth = 64 }>">>) -> tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 32, inde × BitWidth = 32 }>">>
    "func.return"(%0) : (tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 32, inde × BitWidth = 32 }>">>) -> ()
  }) {function_type = (tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 64, inde × BitWidth = 64 }>">>) -> tensor<? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 32, inde × BitWidth = 32 }>">>, sym_name = "sparse_convert_1d_ss"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<2 × 4 × f64>):
    %0 = "sparse_tensor.convert"(%arg0) : (tensor<2 × 4 × f64>) -> tensor<2 × 4 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<2 × 4 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (tensor<2 × 4 × f64>) -> tensor<2 × 4 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_convert_2d"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = sparse<[[0, 0], [1, 6]], [1.000000e+00, 5.000000e+00]> : tensor<8 × 7 × f32>} : () -> tensor<8 × 7 × f32>
    %1 = "sparse_tensor.convert"(%0) : (tensor<8 × 7 × f32>) -> tensor<8 × 7 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%1) : (tensor<8 × 7 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = () -> tensor<8 × 7 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_constant"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<? × ? × ? × f64>):
    %0 = "sparse_tensor.convert"(%arg0) : (tensor<? × ? × ? × f64>) -> tensor<? × ? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>
    "func.return"(%0) : (tensor<? × ? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>) -> ()
  }) {function_type = (tensor<? × ? × ? × f64>) -> tensor<? × ? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>, sym_name = "sparse_convert_3d"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "sparse_tensor.pointers"(%arg0, %0) : (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, index) -> memref<? × inde × >
    "func.return"(%1) : (memref<? × inde × >) -> ()
  }) {function_type = (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × inde × >, sym_name = "sparse_pointers"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 64, inde × BitWidth = 64 }>">>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "sparse_tensor.pointers"(%arg0, %0) : (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 64, inde × BitWidth = 64 }>">>, index) -> memref<? × i64>
    "func.return"(%1) : (memref<? × i64>) -> ()
  }) {function_type = (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 64, inde × BitWidth = 64 }>">>) -> memref<? × i64>, sym_name = "sparse_pointers64"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 32, inde × BitWidth = 32 }>">>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "sparse_tensor.pointers"(%arg0, %0) : (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 32, inde × BitWidth = 32 }>">>, index) -> memref<? × i32>
    "func.return"(%1) : (memref<? × i32>) -> ()
  }) {function_type = (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 32, inde × BitWidth = 32 }>">>) -> memref<? × i32>, sym_name = "sparse_pointers32"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "sparse_tensor.indices"(%arg0, %0) : (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, index) -> memref<? × inde × >
    "func.return"(%1) : (memref<? × inde × >) -> ()
  }) {function_type = (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × inde × >, sym_name = "sparse_indices"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 64, inde × BitWidth = 64 }>">>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "sparse_tensor.indices"(%arg0, %0) : (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 64, inde × BitWidth = 64 }>">>, index) -> memref<? × i64>
    "func.return"(%1) : (memref<? × i64>) -> ()
  }) {function_type = (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 64, inde × BitWidth = 64 }>">>) -> memref<? × i64>, sym_name = "sparse_indices64"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 32, inde × BitWidth = 32 }>">>):
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "sparse_tensor.indices"(%arg0, %0) : (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 32, inde × BitWidth = 32 }>">>, index) -> memref<? × i32>
    "func.return"(%1) : (memref<? × i32>) -> ()
  }) {function_type = (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 32, inde × BitWidth = 32 }>">>) -> memref<? × i32>, sym_name = "sparse_indices32"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "sparse_tensor.values"(%arg0) : (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × f64>
    "func.return"(%0) : (memref<? × f64>) -> ()
  }) {function_type = (tensor<128 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × f64>, sym_name = "sparse_valuesf64"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "sparse_tensor.values"(%arg0) : (tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × f32>
    "func.return"(%0) : (memref<? × f32>) -> ()
  }) {function_type = (tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × f32>, sym_name = "sparse_valuesf32"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × i32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "sparse_tensor.values"(%arg0) : (tensor<128 × i32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × i32>
    "func.return"(%0) : (memref<? × i32>) -> ()
  }) {function_type = (tensor<128 × i32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × i32>, sym_name = "sparse_valuesi32"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × i16, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "sparse_tensor.values"(%arg0) : (tensor<128 × i16, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × i16>
    "func.return"(%0) : (memref<? × i16>) -> ()
  }) {function_type = (tensor<128 × i16, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × i16>, sym_name = "sparse_valuesi16"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × i8, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "sparse_tensor.values"(%arg0) : (tensor<128 × i8, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × i8>
    "func.return"(%0) : (memref<? × i8>) -> ()
  }) {function_type = (tensor<128 × i8, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> memref<? × i8>, sym_name = "sparse_valuesi8"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "sparse_tensor.load"(%arg0) : (tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_reconstruct"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>):
    %0 = "sparse_tensor.load"(%arg0) {hasInserts} : (tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    "func.return"(%0) : (tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> ()
  }) {function_type = (tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, sym_name = "sparse_reconstruct_ins"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, %arg1: memref<? × inde × >, %arg2: f32):
    "sparse_tensor.lex_insert"(%arg0, %arg1, %arg2) : (tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, memref<? × inde × >, f32) -> ()
    "func.return"() : () -> ()
  }) {function_type = (tensor<128 × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, memref<? × inde × >, f32) -> (), sym_name = "sparse_insert"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 8 : index} : () -> index
    %1 = "sparse_tensor.init"(%0, %0) : (index, index) -> tensor<8 × 8 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>
    %2:4 = "sparse_tensor.expand"(%1) : (tensor<8 × 8 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>) -> (memref<? × f64>, memref<? × i1>, memref<? × inde × >, index)
    "func.return"(%2#2) : (memref<? × inde × >) -> ()
  }) {function_type = () -> memref<? × inde × >, sym_name = "sparse_expansion"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<8 × 8 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, %arg1: memref<? × inde × >, %arg2: memref<? × f64>, %arg3: memref<? × i1>, %arg4: memref<? × inde × >, %arg5: index):
    "sparse_tensor.compress"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<8 × 8 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, memref<? × inde × >, memref<? × f64>, memref<? × i1>, memref<? × inde × >, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (tensor<8 × 8 × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, memref<? × inde × >, memref<? × f64>, memref<? × i1>, memref<? × inde × >, index) -> (), sym_name = "sparse_compression"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, %arg1: !llvm<"ptr<i8>">):
    "sparse_tensor.out"(%arg0, %arg1) : (tensor<? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, !llvm<"ptr<i8>">) -> ()
    "func.return"() : () -> ()
  }) {function_type = (tensor<? × ? × f64, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22 ], pointerBitWidth = 0, inde × BitWidth = 0 }>">>, !llvm<"ptr<i8>">) -> (), sym_name = "sparse_out1"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: tensor<? × ? × ? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>, %arg1: !llvm<"ptr<i8>">):
    "sparse_tensor.out"(%arg0, %arg1) : (tensor<? × ? × ? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>, !llvm<"ptr<i8>">) -> ()
    "func.return"() : () -> ()
  }) {function_type = (tensor<? × ? × ? × f32, #sparse_tensor<"encoding<{ dimLevelType = [ \22dense\22, \22compressed\22, \22compressed\22 ], dimOrdering = affine_map<(d0, d1, d2) -> (d2, d0, d1)>, pointerBitWidth = 0, indexBitWidth = 0 }>">>, !llvm<"ptr<i8>">) -> (), sym_name = "sparse_out2"} : () -> ()
}) : () -> ()


] 
-- | main program
def main : IO Unit :=
    -- def astData := gatherASTData o
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "out.txt" str
