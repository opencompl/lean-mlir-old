/-
## MLIR types

This file implements general properties of MLIR types, including decidable
equality of both types and values, Inhabited instances, and concretization of
MLIR types to standard Lean types.

Only the types explicitly laid out in MLIRType are considered here; non-trivial
types like tensors and user-defined types provide similar properties through
the generic type interface.

We require every concrete MLIR type to be Inhabited so that we can keep the
`SSAEnv` untyped and have `SSAEnvE.Get` return default values in case of a
dynamic type mismatch (which is provably impossible but extremely impractical
to use as a guarantee).

Current properly-supported MLIR built-in types:

* Tuple type [(τ₁, ..., τₙ)]: n-ary product of underlying types
* Index type: infinite precision Int
* Generic type: via type interface

Types that need improvements or refinements:

* Function type [τ₁ → τ₂]
  Resolves to program symbols, lacks testing
* Integer types [i1, si32, u64, etc]
  Very poor support for actual operations on FinInt
* Float types [f16, f32, f64, f80, f128]
  Good luck with these
-/

import MLIR.Util.Arith
import MLIR.Util.List
import MLIR.Util.FinInt
import MLIR.Util.KDTensor
import MLIR.AST


import Lean
import Lean.Elab.Term
import Lean.Elab.Exception

open MLIR.AST
open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Parser.Term

section
variable {α σ ε} [δ: Dialect α σ ε]

/-
### Decidable equality for MLIRType
-/

mutual

def MLIRType.eq (τ₁ τ₂: MLIRType δ): Decidable (τ₁ = τ₂) := by
  cases τ₁ <;> cases τ₂
  <;> try (simp; exact inferInstance)
  <;> try apply isFalse MLIRType.noConfusion

private def MLIRType.eqList (l₁ l₂: List (MLIRType δ)): Decidable (l₁ = l₂) :=
  match l₁, l₂ with
  | [], [] => isTrue rfl
  | τ₁::l₁, τ₂::l₂ =>
      match eq τ₁ τ₂, eqList l₁ l₂ with
      | isTrue hτ, isTrue hl => isTrue $ by rw [hτ,hl]
      | isFalse hτ, _ => isFalse fun h => by cases h; cases hτ rfl
      | _, isFalse hl => isFalse fun h => by cases h; cases hl rfl
  | [], _::_ => isFalse List.noConfusion
  | _::_, [] => isFalse List.noConfusion
end

instance: DecidableEq (MLIRType δ) :=
  MLIRType.eq


/-
### Evaluation into concrete Lean types
-/

/- MLIRType used to be a nested inductive type, due to the presence of function and tuple
  types. We have removed this, since we do not need these types. Instead, arguments
  and return values are decoreated with the expected type.

 Note that Recursive functions on nested inductives are
 compiled to well-founded recursion. This prevents it from being reduced by
 the elaborator, so instead we define it manually with the recursor.
 See: https://leanprover.zulipchat.com/#narrow/stream/270676-lean4/topic/reduction.20of.20dependent.20return.20type/near/276044057 -/

@[reducible, simp]
def MLIR.AST.MLIRType.eval: MLIRType δ -> Type
| .float _ => Float
| .int signedness sz=> FinInt sz
| .tensor1d => Tensor1D
| .tensor2d => Tensor2D
| .tensor4d => Tensor4D
| .index => Int
| .undefined _ => Unit
| .extended σ => ε σ
| .erased => Unit


/-
### Properties of evaluated types

The requirements from the type interface allow us to prove that MLIR types have
inhabitants and a decidable equality.
-/

def MLIR.AST.MLIRType.default (τ: MLIRType δ): τ.eval :=
  match τ with
  | .int _ _ => .zero
  | .float _ => 0.0
  | .index => 0
  | .tensor1d => Tensor1D.empty
  | .tensor2d => Tensor2D.empty
  | .tensor4d => Tensor4D.empty
  | .undefined name => ()
  | .extended s => DialectTypeIntf.inhabited s
  | .erased => ()



instance (τ: MLIRType δ): Inhabited τ.eval where
  default := τ.default

def MLIRType.eval.eq {τ: MLIRType δ} (v₁ v₂: τ.eval): Decidable (v₁ = v₂) :=
  match τ with
  | .int _ _ => inferInstance
  | .float _ =>
      -- FIXME: Equality of floats
      if v₁ == v₂ then isTrue sorry else isFalse sorry
  | .tensor1d => Tensor1D.isEq v₁ v₂
  | .tensor2d => Tensor2D.isEq v₁ v₂
  | .tensor4d => Tensor4D.isEq v₁ v₂
  | .index => inferInstance
  | .undefined _ => inferInstance
  | .erased => inferInstance
  | .extended s => DialectTypeIntf.eq s v₁ v₂

instance {τ: MLIRType δ}: DecidableEq τ.eval :=
  MLIRType.eval.eq


def MLIRType.eval.str {τ: MLIRType δ} (v: τ.eval): String :=
  match τ, v with
  | .int .Signless _, v => toString v.toUint
  | .int .Unsigned _, v => toString v.toUint
  | .int .Signed 0, v => "<i0>"
  | .int .Signed (sz+1), v => toString v.toSint
  | .float _, v => toString v
  | .tensor1d, t => toString t
  | .tensor2d, t => toString t
  | .tensor4d, t => toString t
  | .index, v => toString v
  | .undefined _, () => "<undefined>"
  | .erased, () => "<erased>"
  | .extended s, v => DialectTypeIntf.str s v

instance {τ: MLIRType δ}: ToString τ.eval where
  toString := MLIRType.eval.str

end -- of section defining δ
