-- Ideas from: https://www.cs.utah.edu/~regehr/papers/pldi15.pdf

import Lean.Parser
import MLIR.AST
import MLIR.EDSL
import MLIR.Doc
import Lean.Parser.Extra

open MLIR.AST
open MLIR.EDSL
open MLIR.Doc
open Std
open Lean
open Lean.Parser


-- COMBINATORIAL DIALECT
-- ====================
-- https://circt.llvm.org/docs/Dialects/Comb/
namespace circt_comb
  -- | TODO: think about how to do types
  -- def comb_add (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.add"( {{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]
  -- -- | TODO: think about how to do types
  -- def comb_and (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.and"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]
  -- def comb_concat (x: SSAVal) (wx: Int) (y: SSAVal) (wy: Int) : Op := 
  --   let tyx := MLIRTy.int $ wx
  --   let tyy := MLIRTy.int $ wy
  --   let ty := MLIRTy.int $ wx + wy
  --   [mlir_op| "comb.concat"({{ x }}, {{y}}) : ({{tyx}}, {{tyy}}) -> {{ty}}]
  -- def comb_divS (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int $ width
  --   [mlir_op| "comb.divs"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_divU (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int $ width
  --   [mlir_op| "comb.divu"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- -- def comb_extract (x: SSAVal)  (width: Int) (lowBit: Int) : Op := 
  -- --   let ty := MLIRTy.int $ width
  -- --   -- | TODO: add integer attribute
  -- --   [mlir_op| "comb.extract"({{ x }}){ lowBit=10} : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_icmp (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.icmp"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_modS (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.mods"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_modU (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.modu"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_mul (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.mul"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_mux (cond: SSAVal) (trueVal: SSAVal) (falseVal: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.mux"({{ cond }}, {{trueVal}}, {{falseVal}}) : (i1, {{ty}}) -> ({{ty}}) ]

  -- def comb_or (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.or"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_parity (x: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.parity"({{ x }}) : ({{ty}}) -> (i1)]


  -- def comb_sext (x: SSAVal) (width: Int) (outWidth: Int) : Op := 
  --   let tyin := MLIRTy.int width
  --   let tyout := MLIRTy.int outWidth
  --   [mlir_op| "comb.sext"({{ x }}) : ({{tyin}}) -> ({{tyout}})]

  -- def comb_shl (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.shl"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_shrS (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.shrs"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_shrU (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.shru"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_sub (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.sub"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]

  -- def comb_xor (x: SSAVal) (y: SSAVal) (width: Int) : Op := 
  --   let ty := MLIRTy.int width
  --   [mlir_op| "comb.sub"({{ x }}, {{y}}) : ({{ty}}, {{ty}}) -> {{ty}}]
end circt_comb

