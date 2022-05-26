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
* Integer types [i32, etc]
  TODO: Model i32/etc with finite precision, probably restarting from Fin
* Unsigned finite integer types [u32, etc]
  TODO: Model u32/etc with lean's Uint{8/16/32/64} or restart with Fin
* Float types [f16, f32, f64, f80, f128]
  Good luck with these
-/

import MLIR.Util.Arith
import MLIR.Util.List
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
    (fun bitsize => Int)
    -- .float
    (fun bitsize => Float)
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
  | .int _ => 0
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
  | .int _ => inferInstance
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
  | .int _, v => toString v
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


/-
## Decidable equality for TensorElem
-/

mutual
def TensorElem.eq (e₁ e₂: TensorElem): Decidable (e₁ = e₂) := by
  cases e₁ <;> cases e₂
  <;> try (simp; exact inferInstance)
  <;> try apply isFalse TensorElem.noConfusion

  case float.float f₁ f₂ =>
    -- FIXME: We shouldn't have DecidableEq on floats o(x_x)o
    exact if f₁ == f₂ then isTrue sorry else isFalse sorry

  case nested.nested l₁ l₂ =>
    match eqList l₁ l₂ with
    | isTrue h => exact isTrue $ by rw [h]
    | isFalse h => exact isFalse fun h' => by cases h'; cases h rfl

private def TensorElem.eqList (l₁ l₂: List TensorElem): Decidable (l₁ = l₂) :=
  match l₁, l₂ with
  | [], [] => isTrue rfl
  | e₁::l₁, e₂::l₂ =>
      match eq e₁ e₂, eqList l₁ l₂ with
      | isTrue hτ, isTrue hl => isTrue $ by rw [hτ,hl]
      | isFalse hτ, _ => isFalse fun h => by cases h; cases hτ rfl
      | _, isFalse hl => isFalse fun h => by cases h; cases hl rfl
  | [], _::_ => isFalse List.noConfusion
  | _::_, [] => isFalse List.noConfusion
end

instance: DecidableEq TensorElem :=
  TensorElem.eq


/-
## Shape inference on TensorElem

This section defines shape verification and shape inference for TensorElem
(tensor literals), *excluding the case of uniform tensor literals*. The shape
inference is proven correct, and the `flatten` method is defined that exports
the tensor literal to a flat array suitable for use in a `RankedTensor`.

`RankedTensor` provides the functions that actually turn tensor literals into
ranked tensors and properly handle uniform tensor literals.

TODO: Integrate TensorElem invariants into the verifier
-/

def shape_prod: List Nat → Nat :=
  List.foldr (·*·) 1

theorem shape_prod_nil: shape_prod (0::l) = 0 := by
  induction l <;> simp [shape_prod, List.foldr]

namespace MLIR.AST.TensorElem

-- Check whether a tensor literal matches a concrete shape
def hasShape: TensorElem → List Nat → Bool
  | TensorElem.empty, _ =>
      false
  | TensorElem.int _, [] =>
      true
  | TensorElem.int _, _::_ =>
      false
  | TensorElem.bool _, [] =>
      true
  | TensorElem.bool _, _::_ =>
      false
  | TensorElem.float _, [] =>
      true
  | TensorElem.float _, _::_ =>
      false
  | TensorElem.nested l, rank::size =>
      l.length = rank ∧ l.all (hasShape . size)
  | TensorElem.nested _, [] =>
      false

-- Check whether a tensor literal has a particular data type
def hasType: TensorElem → MLIRType δ → Bool
  | TensorElem.int _, .int _ =>
      -- TODO: Check bounds
      true
  | TensorElem.bool _, .int 1 =>
      true
  | TensorElem.float _, .float _ =>
      true
  | TensorElem.nested [], τ =>
      true
  | TensorElem.nested (e::l), τ =>
      e.hasType τ ∧ (TensorElem.nested l).hasType τ
  | _, _ =>
      false

def hasType_list_1 {l} {τ: MLIRType δ}:
    hasType (.nested l) τ → l.all (hasType . τ) := by
  induction l; simp
  case cons e l ih =>
    simp [hasType, List.all_cons]
    intro h
    simp [h.1]
    apply ih h.2

def hasType_list_2 {l} {τ: MLIRType δ}:
    l.all (hasType . τ) → hasType (.nested l) τ := by
  induction l; simp [hasType]
  case cons e l ih =>
    simp [hasType, List.all_cons]
    intro h
    simp [h.1]
    apply ih h.2

def mapWithType {α} {τ: MLIRType δ} l (f: (e: TensorElem) → (h: e.hasType τ) → α)
    (h: hasType (TensorElem.nested l) τ): List α :=
  match l, h with
  | [], h =>
      []
  | e::l, h =>
      let h₁ := (by simp [hasType] at h; apply h.1)
      let h₂ := (by simp [hasType] at h; apply h.2)
      f e h₁ :: mapWithType l f h₂


-- Shape inference function; this determines the unique shape that we allow a
-- non-uniform tensor can have (hasShape is more liberal with empty lists, but
-- the MLIR compiler is not)
def inferredShape: TensorElem → Option (List Nat)
  | TensorElem.empty =>
      none
  | TensorElem.int _ =>
      some []
  | TensorElem.bool _ =>
      some []
  | TensorElem.float _ =>
      some []
  | TensorElem.nested [] =>
      some [0]
  | TensorElem.nested (e::l) =>
      Option.bind (inferredShape e) fun s1 =>
      Option.bind (inferredShape (TensorElem.nested l)) fun s2 =>
      match s2 with
      | [] => none /- impossible -/
      | head :: tail =>
        if s1 = tail then some ((head+1) :: s1) else none

-- First let's prove the list case equivalent to a more readable form

theorem inferredShape_cons: ∀ head tail s_head s_tail,
    inferredShape (TensorElem.nested tail) = some (s_head :: s_tail) →
    inferredShape head = some s_tail →
    inferredShape (TensorElem.nested (head :: tail)) =
      some ((s_head+1) :: s_tail) := by
  intros head tail s_head s_tail H1 H2
  simp [inferredShape, H2, Option.bind, H1];

theorem inferredShape_cons_inv: ∀ {head tail s_head s_tail},
    inferredShape (TensorElem.nested (head::tail)) = some (s_head::s_tail) →
    s_head > 0 ∧
    inferredShape head = some s_tail ∧
    inferredShape (TensorElem.nested tail) = some ((s_head-1) :: s_tail) := by
  intros head tail s_head s_tail
  simp [inferredShape]
  cases inferredShape head <;> simp [Option.bind]
  case some head_shape =>
  cases inferredShape (TensorElem.nested tail) <;> simp [Option.bind]
  case some tail_shape =>
  cases tail_shape <;> simp
  case cons s_head' s_tail' =>
  apply dite (head_shape = s_tail')
  . intros Heq; rw [Heq]; simp
    intros H; rw [←H.1, Nat.add_sub_self_right, ←H.2]
    exact ⟨by simp_arith, rfl, rfl, rfl⟩
  . intros Hne; simp [Hne]

theorem inferredShape_list {l head tail}:
    inferredShape (TensorElem.nested l) = some (head::tail) →
    head = l.length ∧ l.all (inferredShape . = some tail) := by
  revert head tail; induction l <;> simp
  case nil =>
    intros head tile H; simp [inferredShape, List.all_nil] at *; simp [H.1]
  case cons head tail ih =>
    intros s_head s_tail H
    let H' := inferredShape_cons_inv H
    specialize (ih H'.2.2)
    constructor
    . simp [←ih.1, Nat.succ_eq_add_one, Nat.minus_plus_one H'.1]
    . simp [List.all_cons]
      constructor; exact H'.2.1; exact ih.2

theorem inferredShape_list_to_cons {l s}:
    inferredShape (TensorElem.nested l) = some s →
    ∃ tail, s = l.length :: tail := by
  cases s <;> simp [inferredShape]
  case nil =>
    cases l <;> simp [inferredShape]
    case cons head tail =>
      cases inferredShape head <;> simp [Option.bind]
      cases inferredShape (TensorElem.nested tail) <;> simp [Option.bind]
      case some.some s1 s2 =>
        cases s2 <;> simp
        case cons s2_head s2_tail =>
          apply dite (s1 = s2_tail) <;> intros H <;> simp [H]
  case cons s_head s_tail =>
    intro H
    let H' := inferredShape_list H
    apply Exists.intro s_tail
    exact ⟨H'.1, rfl⟩

-- We can now show that the shape inference function is correct

theorem hasShape_inferredShape_1:
  ∀ (e: TensorElem) (shape: List Nat),
    e.inferredShape = some shape → e.hasShape shape := by
  intro e
  -- Cannot use the [induction] tactic because TensorElem is a nested inductive
  -- and the tactic only supports recursors with a single motive
  apply @TensorElem.recOn
    (motive_1 := fun e =>
      ∀s, e.inferredShape = some s → e.hasShape s)
    (motive_2 := fun l =>
      ∀s, l.all (TensorElem.inferredShape . = some s) →
        l.all (TensorElem.hasShape . s))
  case int =>
    intros _ s H; cases s <;> simp [inferredShape, hasShape] at *
  case bool =>
    intros _ s H; cases s <;> simp [inferredShape, hasShape] at *
  case float =>
    intros _ s H; cases s <;> simp [inferredShape, hasShape] at *
  case nested =>
    intros l motive_2 s H
    let H' := inferredShape_list_to_cons H
    cases H'; case intro s_tail Hs =>
      rw [Hs]; rw [Hs] at H; clear Hs H'
      let H' := inferredShape_list H
      simp [hasShape, motive_2 _ H'.2]
  case empty =>
    simp [inferredShape]
  case nil =>
    intros s H; simp [List.all_nil]
  case cons =>
    intros head tail motive_1 ih s H; simp [List.all_cons] at *
    simp [motive_1 _ H.1, ih _ H.2]

end MLIR.AST.TensorElem


/-
### Tools for generation of ranked tensors
TODO: Consider a different type KnownRankedTensor that we could project to if
TODO| we have known dimensions. Everything is conditioned by DimList.known...
-/

@[simp]
def shape_refines: List Nat → List Dimension → Bool
  | [], [] => true
  | size::shape, Dimension.Unknown::dim => shape_refines shape dim
  | size::shape, Dimension.Known d::dim => size = d && shape_refines shape dim
  | (_::_), [] => false
  | [], (_::_) => false

@[inline]
def DimList := List Dimension

deriving instance DecidableEq for DimList

@[simp]
def DimList.prod: DimList → Nat
  | [] => 1
  | Dimension.Known n :: D => n * prod D
  | Dimension.Unknown :: _ => 0

@[simp]
def DimList.project: DimList → List Nat
  | [] => []
  | Dimension.Known n :: D => n :: project D
  | Dimension.Unknown :: D => project D

@[simp]
def DimList.known: DimList → Bool
  | [] => true
  | Dimension.Known n :: D => known D
  | Dimension.Unknown :: _ => false

@[simp]
def DimList.default_refinement: DimList → List Nat
  | [] => []
  | Dimension.Known n :: D => n :: default_refinement D
  | Dimension.Unknown :: D => 0 :: default_refinement D

theorem dim_known_project_refines {D: DimList}:
    D.known → shape_refines D.project D := by
  intros h <;> induction D <;> simp
  case cons head tail ih =>
    cases head <;> simp at *; apply (ih h)

theorem dim_known_refines_inv {D: DimList} {S: List Nat}:
    D.known → shape_refines S D → D = S.map Dimension.Known := by
  intros Hknown; revert S; induction D <;> intros S Hrefines
  case nil =>
    cases S; simp [List.map]; simp at Hrefines
  case cons head tail ih =>
    cases S; simp at Hrefines
    simp [List.map]; cases head <;> simp at *
    rw [Hrefines.1, ←ih Hknown]; apply Hrefines.2

theorem dim_known_project_eq {D: DimList}:
    D.known → shape_refines S D → D.project = S := by
  intros Hknown Hrefines
  rw [dim_known_refines_inv Hknown Hrefines]
  clear D Hknown Hrefines
  induction S <;> simp; assumption

theorem dim_known_prod_refines {D: DimList}:
    D.known → shape_refines S D → shape_prod S = D.prod := by
  intros Hknown; revert S; induction D <;> intros S Hrefines <;> simp
  case nil =>
    cases S; simp; simp at Hrefines
  case cons head tail ih =>
    cases S; simp at Hrefines
    cases head <;> simp at *
    rw [←Hrefines.1, ←ih Hknown Hrefines.2]
    simp [shape_prod, List.foldr]

theorem dim_known_prod (D: DimList):
    D.known → shape_prod D.project = D.prod :=
  fun Hknown =>
    dim_known_prod_refines Hknown (dim_known_project_refines Hknown)

theorem default_refinement_refines (D: DimList):
    shape_refines D.default_refinement D := by
  induction D <;> simp
  case cons head _ ih =>
    cases head <;> simp <;> apply ih


namespace MLIR.AST.TensorElem

section
variable {α σ ε} {δ: Dialect α σ ε}

def flatten {τ: MLIRType δ} (e: TensorElem) (h: e.hasType τ): List τ.eval :=
  match e, τ with
  | TensorElem.int i, .int _ =>
      [i]
  | TensorElem.bool b, .int _ =>
      [if b then 1 else 0]
  | TensorElem.float f, .float _ =>
      [f]
  | TensorElem.nested [], _ =>
      []
  | TensorElem.nested (e::l), τ =>
      let h₁ := (by simp [hasType] at h; apply h.1)
      let h₂ := (by simp [hasType] at h; apply h.2)
      flatten e h₁ ++ flatten (TensorElem.nested l) h₂
  | _, _ =>
      [] -- TODO: Prove impossible

-- Once again, we prove a more friendly version of the list case first

theorem flatten_list {τ: MLIRType δ} (l: List TensorElem) (h: hasType (.nested l) τ):
    flatten (.nested l) h = (mapWithType l flatten h).join := by
  revert h
  induction l <;> intros h
  case nil =>
    simp [flatten, mapWithType, List.join]
  case cons _ _ ih =>
    simp [flatten, mapWithType, List.join, ih]

theorem flatten_size {τ: MLIRType δ} (e: TensorElem) (shape: List Nat):
    e.hasShape shape → (h: e.hasType τ) → (e.flatten h).length = shape_prod shape := by
  revert shape
  apply @TensorElem.recOn
    (motive_1 := fun e =>
      ∀s, e.hasShape s → (h: e.hasType τ) → (e.flatten h).length = shape_prod s)
    (motive_2 := fun l =>
      ∀s, l.all (TensorElem.hasShape . s) → (h: l.all (hasType . τ)) →
        (mapWithType l flatten (hasType_list_2 h)).join.length = l.length * shape_prod s)
    <;> simp <;> clear e
  case int =>
    intros i s Hshape Htype;
    cases τ <;> simp [hasType] at Htype
    cases s <;> simp [flatten, hasShape] at *
  case float =>
    intros i s Hshape Htype;
    cases τ <;> simp [hasType] at Htype
    cases s <;> simp [flatten, hasShape] at *
  case bool =>
    intros i s Hshape Htype;
    cases τ <;> simp [hasType] at Htype
    cases s <;> simp [flatten, hasShape] at *
  case nested =>
    intros l motive_2 s Hshape Htype
    cases s <;> simp [hasShape] at Hshape
    case cons s_head s_tail =>
    simp [TensorElem.flatten_list, shape_prod, List.foldr]
    simp [motive_2 s_tail Hshape.2 (hasType_list_1 Htype)]
    simp [shape_prod, Nat.mul_comm, Hshape.1]
  case empty =>
    intros s Hshape Htype
    simp [hasType] at Htype
  case nil =>
    intros _ Htype
    simp [mapWithType, List.join]
  case cons =>
    intros head tail motive_1 IH2 s Hshape Htype
    simp [List.map, List.join]
    rw [Nat.add_comm]
    simp [Nat.succ_eq_add_one, Nat.right_distrib]
    simp [List.all_cons] at Hshape
    simp [List.all_cons] at Htype
    simp [IH2 s Hshape.2 Htype.2]
    rw [motive_1 s Hshape.1 Htype.1]

inductive rankCompatibleWith (e: TensorElem) (D: DimList): MLIRType δ → Type :=
  | UniformInt (i: Int) bitsize:
      -- TODO: Check range of uniform tensor value
      e = TensorElem.int i →
      e.rankCompatibleWith D (.int bitsize)
  | UniformBool (b: Bool):
      e = TensorElem.bool b →
      e.rankCompatibleWith D (.int 1)
  | UniformFloat (f: Float) bitsize:
      -- TODO: Check range of uniform tensor value
      e = TensorElem.float f →
      e.rankCompatibleWith D (.float bitsize)
  | HasShape s τ:
      e.hasShape s →
      shape_refines s D →
      e.rankCompatibleWith D τ

end
end MLIR.AST.TensorElem
