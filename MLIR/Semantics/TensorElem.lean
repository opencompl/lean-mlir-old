/-
## Literal tensors

This file provides utilities to validate `TensorElem` objects (literal tensors)
with shape inference and shape/type verification. It defines a `TensorLiteral`
structure which extends `TensorElem` with invariants and can be extracted into
a concrete `builtin.tensor` value.
-/

import MLIR.AST
import MLIR.Semantics.Types
import MLIR.Util.List

open MLIR.AST

/-
### Decidable equality
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
### Shape inference

This section defines shape verification and shape inference for TensorElem
(tensor literals), *excluding the case of uniform tensor literals*. The shape
inference is proven correct, and the `flatten` method is defined that exports
the tensor literal to a flat array suitable for use in a `RankedTensor`.

`RankedTensor` provides the functions that actually turn tensor literals into
ranked tensors and properly handle uniform tensor literals.

TODO: Integrate TensorElem invariants into the verifier
-/

namespace MLIR.AST.TensorElem

def shapeProd: List Nat → Nat :=
  List.foldr (·*·) 1

theorem shape_prod_nil: shapeProd (0::l) = 0 := by
  induction l <;> simp [shapeProd, List.foldr]

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
def hasType: TensorElem → MLIRTy → Bool
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

theorem hasType_list_1 {l} {τ: MLIRTy}:
    hasType (.nested l) τ → l.all (hasType . τ) := by
  induction l; simp
  case cons e l ih =>
    simp [hasType, List.all_cons]
    intro h
    simp [h.1]
    apply ih h.2

theorem hasType_list_2 {l} {τ: MLIRTy}:
    l.all (hasType . τ) → hasType (.nested l) τ := by
  induction l; simp [hasType]
  case cons e l ih =>
    simp [hasType, List.all_cons]
    intro h
    simp [h.1]
    apply ih h.2

def mapWithType {τ: MLIRTy} l (f: (e: TensorElem) → (h: e.hasType τ) → α)
    (h: hasType (TensorElem.nested l) τ): List α :=
  match l, h with
  | [], h =>
      []
  | e::l, h =>
      let h₁ := (by simp [hasType] at h; apply h.1)
      let h₂ := (by simp [hasType] at h; apply h.2)
      f e h₁ :: mapWithType l f h₂


-- Shape inference function; this determines the unique shape that we allow a
-- non-uniform tensor can have (`hasShape` is more liberal with empty lists,
-- but the MLIR compiler is not)
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
  | TensorElem.nested (e::l) => do
      let s1 ← inferredShape e
      let s2 ← inferredShape (.nested l)
      match s2 with
      | [] => none /- impossible -/
      | head :: tail => if s1 = tail then some ((head+1) :: tail) else none

-- First let's prove the list case equivalent to a more readable form

theorem inferredShape_cons: ∀ head tail s_head s_tail,
    inferredShape (.nested tail) = some (s_head :: s_tail) →
    inferredShape head = some s_tail →
    inferredShape (.nested (head :: tail)) =
      some ((s_head+1) :: s_tail) := by
  intros head tail s_head s_tail H1 H2
  simp [inferredShape, H2, bind, Option.bind, H1]

theorem inferredShape_cons_inv: ∀ {head tail s_head s_tail},
    inferredShape (.nested (head::tail)) = some (s_head::s_tail) →
    s_head > 0 ∧
    inferredShape head = some s_tail ∧
    inferredShape (.nested tail) = some ((s_head-1) :: s_tail) := by
  intros head tail s_head s_tail
  simp [inferredShape]
  cases inferredShape head <;> simp [bind, Option.bind]
  case some head_shape =>
  cases inferredShape (.nested tail) <;> simp [Option.bind]
  case some tail_shape =>
  cases tail_shape <;> simp
  case cons s_head' s_tail' =>
  apply dite (head_shape = s_tail')
  . intros Heq; rw [Heq]; simp
    intros H; rw [←H.1, Nat.add_sub_self_right, ←H.2]
    exact ⟨by simp_arith, rfl, rfl, rfl⟩
  . intros Hne; simp [Hne]

theorem inferredShape_list {l head tail}:
    inferredShape (.nested l) = some (head::tail) →
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
    inferredShape (.nested l) = some s →
    ∃ tail, s = l.length :: tail := by
  cases s <;> simp [inferredShape]
  case nil =>
    cases l <;> simp [inferredShape]
    case cons head tail =>
      cases inferredShape head <;> simp [bind, Option.bind]
      cases inferredShape (.nested tail) <;> simp [Option.bind]
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

theorem hasShape_inferredShape:
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
-/

@[inline]
def DimList := List Dimension

namespace DimList

deriving instance DecidableEq for DimList

@[simp]
def shapeRefines: List Nat → DimList → Bool
  | [], [] => true
  | size::shape, .Unknown::dim => shapeRefines shape dim
  | size::shape, .Known d::dim => size = d && shapeRefines shape dim
  | (_::_), [] => false
  | [], (_::_) => false

@[simp]
def prod: DimList → Nat
  | [] => 1
  | .Known n :: D => n * prod D
  | .Unknown :: _ => 0

@[simp]
def project: DimList → List Nat
  | [] => []
  | .Known n :: D => n :: project D
  | .Unknown :: D => project D

@[simp]
def known: DimList → Bool
  | [] => true
  | .Known n :: D => known D
  | .Unknown :: _ => false

@[simp]
def defaultRefinement: DimList → List Nat
  | [] => []
  | .Known n :: D => n :: defaultRefinement D
  | .Unknown :: D => 0 :: defaultRefinement D

theorem dim_known_project_refines {D: DimList}:
    D.known → shapeRefines D.project D := by
  intros h <;> induction D <;> simp
  case cons head tail ih =>
    cases head <;> simp at *; apply (ih h)

theorem dim_known_refines_inv {D: DimList} {S: List Nat}:
    D.known → shapeRefines S D → D = S.map Dimension.Known := by
  intros Hknown; revert S; induction D <;> intros S Hrefines
  case nil =>
    cases S; simp [List.map]; simp at Hrefines
  case cons head tail ih =>
    cases S; simp at Hrefines
    simp [List.map]; cases head <;> simp at *
    rw [Hrefines.1, ←ih Hknown]; apply Hrefines.2

theorem dim_known_project_eq {D: DimList}:
    D.known → shapeRefines S D → D.project = S := by
  intros Hknown Hrefines
  rw [dim_known_refines_inv Hknown Hrefines]
  clear D Hknown Hrefines
  induction S <;> simp; assumption

theorem dim_known_prod_refines {D: DimList}:
    D.known → shapeRefines S D → TensorElem.shapeProd S = D.prod := by
  intros Hknown; revert S; induction D <;> intros S Hrefines <;> simp
  case nil =>
    cases S; simp; simp at Hrefines
  case cons head tail ih =>
    cases S; simp at Hrefines
    cases head <;> simp at *
    rw [←Hrefines.1, ←ih Hknown Hrefines.2]
    simp [TensorElem.shapeProd, List.foldr]

theorem dim_known_prod (D: DimList):
    D.known → TensorElem.shapeProd D.project = D.prod :=
  fun Hknown =>
    dim_known_prod_refines Hknown (dim_known_project_refines Hknown)

theorem defaultRefinement_refines (D: DimList):
    shapeRefines D.defaultRefinement D := by
  induction D <;> simp
  case cons head _ ih =>
    cases head <;> simp <;> apply ih

end DimList

namespace MLIR.AST.TensorElem

def flatten {τ: MLIRTy} (e: TensorElem) (h: e.hasType τ): List τ.eval :=
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
      [] /- impossible -/

-- Once again, we prove a more friendly version of the list case first

theorem flatten_list {τ: MLIRTy} (l: List TensorElem) (h: hasType (.nested l) τ):
    flatten (.nested l) h = (mapWithType l flatten h).join := by
  revert h
  induction l <;> intros h
  case nil =>
    simp [flatten, mapWithType, List.join]
  case cons _ _ ih =>
    simp [flatten, mapWithType, List.join, ih]

theorem flatten_size {τ: MLIRTy} (e: TensorElem) (shape: List Nat):
    e.hasShape shape → (h: e.hasType τ) → (e.flatten h).length = shapeProd shape := by
  revert shape
  apply @TensorElem.recOn
    (motive_1 := fun e =>
      ∀s, e.hasShape s → (h: e.hasType τ) → (e.flatten h).length = shapeProd s)
    (motive_2 := fun l =>
      ∀s, l.all (TensorElem.hasShape . s) → (h: l.all (hasType . τ)) →
        (mapWithType l flatten (hasType_list_2 h)).join.length = l.length * shapeProd s)
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
    simp [TensorElem.flatten_list, shapeProd, List.foldr]
    simp [motive_2 s_tail Hshape.2 (hasType_list_1 Htype)]
    simp [shapeProd, Nat.mul_comm, Hshape.1]
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

inductive rankCompatibleWith (e: TensorElem) (D: DimList): MLIRTy → Type :=
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
      D.shapeRefines s →
      e.rankCompatibleWith D τ

end MLIR.AST.TensorElem


/-
### `TensorLiteral` type

The `TensorLiteral` bundles a `TensorElem` with all the elements required for
the generation of a flat value array and (later) a `RankedTensor`.
-/

structure TensorLiteral (D: DimList) (τ: MLIRTy) where
  elem: TensorElem
  h_type: elem.hasType τ
  h_rank: elem.rankCompatibleWith D τ
