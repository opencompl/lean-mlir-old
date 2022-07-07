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
import MLIR.Semantics.Fitree
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

  case fn.fn a₁ b₁ a₂ b₂ =>
    match eq a₁ a₂, eq b₁ b₂ with
    | isTrue ha, isTrue hb => exact isTrue $ by rw [ha, hb]
    | isFalse ha, _ => exact isFalse fun h => by cases h; cases ha rfl
    | _, isFalse hb => exact isFalse fun h => by cases h; cases hb rfl

  case tuple.tuple l₁ l₂ =>
    match eqList l₁ l₂ with
    | isTrue h => exact isTrue $ by rw [h]
    | isFalse h => exact isFalse fun h' => by cases h'; cases h rfl

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

/- MLIRType is a nested inductive type. Recursive functions on such types are
   compiled to well-founded recursion. This prevents it from being reduced by
   the elaborator, so instead we define it manually with the recursor.
   See: https://leanprover.zulipchat.com/#narrow/stream/270676-lean4/topic/reduction.20of.20dependent.20return.20type/near/276044057 -/

@[reducible, simp_itree]
def MLIR.AST.MLIRType.eval (τ: MLIRType δ): Type :=
  MLIRType.recOn τ
    (motive_1 := fun _ => Type) -- MLIRType
    (motive_2 := fun _ => Type) -- List MLIRType
    -- .fn (the only functions we can materialize are symbols)
    (fun τ₁ τ₂ eval_τ₁ eval_τ₂ => String)
    -- .int
    (fun sgn sz => FinInt sz)
    -- .float
    (fun sz => Float)
    -- .index
    Nat
    -- .tuple [Mapping motive_2 to motive_1]
    (fun _ ih => ih)
    -- .undefined
    (fun name => Unit)
    -- .generic
    ε
    -- [] (in .tuple)
    Unit
    -- (τ::l) (in .tuple)
    (fun τ l eval_τ eval_l =>
      match l with
      | [] => eval_τ
      | _  => eval_τ × eval_l)


/-
### Properties of evaluated types

The requirements from the type interface allow us to prove that MLIR types have
inhabitants and a decidable equality.
-/

def MLIR.AST.MLIRType.default (τ: MLIRType δ): τ.eval :=
  match τ with
  | .fn τ₁ τ₂ => ""
  | .int _ _ => .zero
  | .float _ => 0.0
  | .index => 0
  | .tuple [] => ()
  | .tuple [τ] => τ.default
  | .tuple (τ₁::τ₂::l) => (τ₁.default, default $ .tuple (τ₂::l))
  | .undefined name => ()
  | .extended s => DialectTypeIntf.inhabited s

instance (τ: MLIRType δ): Inhabited τ.eval where
  default := τ.default

def MLIRType.eval.eq {τ: MLIRType δ} (v₁ v₂: τ.eval): Decidable (v₁ = v₂) :=
  match τ with
  | .fn τ₁ τ₂ => inferInstance
  | .int _ _ => inferInstance
  | .float _ =>
      -- FIXME: Equality of floats
      if v₁ == v₂ then isTrue sorry else isFalse sorry
  | .index => inferInstance
  | .tuple [] => inferInstance
  | .tuple [τ] => @eq τ v₁ v₂
  | .tuple (τ₁::τ₂::τs) =>
      let (v₁, l₁) := v₁
      let (v₂, l₂) := v₂
      match eq v₁ v₂, @eq (.tuple (τ₂::τs)) l₁ l₂ with
      | isTrue h₁, isTrue h₂ => isTrue $ by rw [h₁,h₂]
      | isFalse h₁, _ => isFalse fun h => by cases h; cases h₁ rfl
      | _, isFalse h₂ => isFalse fun h => by cases h; cases h₂ rfl
  | .undefined _ => inferInstance
  | .extended s => DialectTypeIntf.eq s v₁ v₂

instance {τ: MLIRType δ}: DecidableEq τ.eval :=
  MLIRType.eval.eq

def MLIRType.eval.str {τ: MLIRType δ} (v: τ.eval): String :=
  match τ, v with
  | .fn τ₁ τ₂, v => v
  | .int .Signless _, v => toString v.toUint
  | .int .Unsigned _, v => toString v.toUint
  | .int .Signed 0, v => "<i0>"
  | .int .Signed (sz+1), v => toString v.toSint
  | .float _, v => toString v
  | .index, v => toString v
  | .tuple [], v => "()"
  | .tuple [τ], v => "(" ++ (@str τ v).drop 1
  | .tuple (τ₁::τ₂::τs), (v,vs) =>
    "(" ++ str v ++ (@str (.tuple (τ₂::τs)) vs).drop 1
  | .undefined _, () => "<undefined>"
  | .extended s, v => DialectTypeIntf.str s v

instance {τ: MLIRType δ}: ToString τ.eval where
  toString := MLIRType.eval.str

end -- of section defining δ

section coe_eq
variable [δ₁: Dialect α₁ σ₁ ε₁] [δ₂: Dialect α₂ σ₂ ε₂] [c: CoeDialect δ₁ δ₂]


private theorem tuple_equal_inv 
  (a a': A) (b b': B) (AEQ: a = a')  (BEQ: b = b'): (a, b) = (a', b') := by {
 rewrite [AEQ, BEQ];
 rfl;
}

mutual 
theorem coe_type_tuple_eval_eq : 
   MLIRType.eval (MLIRType.tuple τs) = MLIRType.eval (coeMLIRType (c := c) (MLIRType.tuple τs)) := match τs with 
  | [] => by {
     simp [MLIRType.eval];
     simp [coeMLIRType];
     simp [coeMLIRTypeList];
  }
  | τ::τs' => by {
    simp [coeMLIRType];
    simp [MLIRType.eval];
    simp [coeMLIRTypeList];
    sorry
  }
  


theorem coe_type_eval_eq(τ: MLIRType δ₁):
   MLIRType.eval τ = MLIRType.eval (coeMLIRType (δ₁ := δ₁) (δ₂ := δ₂) τ) := 
  match H: τ with 
  | .fn τ1 τ2 => by {
       simp [MLIRType.eval, coeMLIRType];
    }
  | .int sgn val => by {
       simp [MLIRType.eval, coeMLIRType];
  }
  | .float n => by {
       simp [MLIRType.eval, coeMLIRType];
  }
  | .index => by {
       simp [MLIRType.eval, coeMLIRType];
  }
  | .tuple τs => by {
      apply coe_type_tuple_eval_eq;
  }
  | .undefined n => by {
      simp [MLIRType.eval, coeMLIRType];
  }
  | .extended s => by {
   simp [MLIRType.eval];
   simp [coeMLIRType];
   rewrite [<- c.coe_ε_well_defined];
   simp;
  }
end 

