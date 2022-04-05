/-
## MLIR types

This file implements support for builtin MLIR types, as well as conversion to
concrete Lean types and values.

In order to support the untyped SSA environment (which doesn't require name
definitions/uses to match as this is proven later), every concrete MLIR type
should be Inhabited so that `SSAEnvE.Get` can pretend to return default values.

Current properly-supported MLIR built-in types:

* Function type [τ₁ → τ₂]: Lean function
* Tuple type [(τ₁, ..., τₙ)]: n-ary product of underlying types
* Tensors [tensor<DxDx?xDx?xτ>]: RankedTensor type

Types that need improvements or refinements:

* Unsigned finite integer types [u32, etc]
  TODO: Model u32/etc with lean's Uint{8/16/32/64} or restart with Fin
* Integer types [i32, etc]: currently use Int
  TODO: Model i32/etc properly, probably restarting from Fin
* Float types [f16, f32, f64, f80, f128]. todo
  No idea how to model floats of different precisions in Lean?
* Vectors: could expand on tensors. todo
* Unranked tensors: todo
* User types: should use a typeclass
  TODO: Model user types
-/

import MLIRSemantics.Util.Arith
import MLIRSemantics.Util.List
import MLIRSemantics.Fitree

import MLIR.AST
open MLIR.AST

def shape_prod: List Nat → Nat :=
  List.foldr (·*·) 1

theorem shape_prod_nil: shape_prod (0::l) = 0 := by
  induction l <;> simp [shape_prod, List.foldr]

instance: OfNat Dimension (n: Nat) where
  ofNat := Dimension.Known n


/-
## Shape inference on literal tensors

This section defines shape verification and shape inference for TensorElem
(tensor literals), *excluding the case of uniform tensor literals*. The shape
inference is proven correct, and the `flatten` method is defined that exports
the tensor literal to a flat array suitable for use in a `RankedTensor`.

`RankedTensor` provides the functions that actually turn tensor literals into
ranked tensors and properly handle uniform tensor literals.

TODO: Integrate TensorElem invariants into the verifier
-/

namespace MLIR.AST.TensorElem

-- Check whether a tensor literal matches a concrete shape
def hasShape: TensorElem → List Nat → Bool
  | TensorElem.int _, [] =>
      true
  | TensorElem.nested l, rank::size =>
      l.length = rank ∧ l.all (hasShape . size)
  | TensorElem.int _, _::_ =>
      false
  | TensorElem.nested _, [] =>
      false

-- Shape inference function; this determines the unique shape that we allow a
-- non-uniform tensor can have (hasShape is more liberal with empty lists, but
-- the MLIR compiler is not)
def inferredShape: TensorElem → Option (List Nat)
  | TensorElem.int _ =>
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
  case nested =>
    intros l motive_2 s H
    let H' := inferredShape_list_to_cons H
    cases H'; case intro s_tail Hs =>
      rw [Hs]; rw [Hs] at H; clear Hs H'
      let H' := inferredShape_list H
      simp [hasShape, motive_2 _ H'.2]
  case nil =>
    intros s H; simp [List.all_nil]
  case cons =>
    intros head tail motive_1 ih s H; simp [List.all_cons] at *
    simp [motive_1 _ H.1, ih _ H.2]

-- Which allows us to flatten the term into a single array for a RankedTensor

def flatten: TensorElem → List Int
  | TensorElem.int i =>
      [i]
  | TensorElem.nested [] =>
      []
  | TensorElem.nested (e::l) =>
      flatten e ++ flatten (TensorElem.nested l)

-- Once again, we prove a more friendly version of the list case first

theorem flatten_list:
  ∀ (l: List TensorElem),
    flatten (TensorElem.nested l) = (l.map flatten).join := by
  intros l; induction l <;> simp
  case cons _ _ ih =>
    simp [flatten, List.map, List.join, ih]

theorem flatten_size (e: TensorElem) (shape: List Nat):
    e.hasShape shape → e.flatten.length = shape_prod shape := by
  revert shape
  apply @TensorElem.recOn
    (motive_1 := fun e =>
      ∀s, e.hasShape s → e.flatten.length = shape_prod s)
    (motive_2 := fun l =>
      ∀s, l.all (TensorElem.hasShape . s) →
        (l.map TensorElem.flatten).join.length = l.length * shape_prod s)
    <;> simp <;> clear e
  case int =>
    intros _ s H;
    cases s <;> simp [TensorElem.flatten, TensorElem.hasShape] at *
  case nested =>
    intros l motive_2 s H
    cases s <;> simp [TensorElem.hasShape] at H
    case cons s_head s_tail =>
    simp [TensorElem.flatten_list, shape_prod, List.foldr]
    simp [motive_2 s_tail H.2, shape_prod, Nat.mul_comm, H.1]
  case cons =>
    intros head tail motive_1 IH2 s H
    simp [List.map, List.join, Nat.add_comm]
    simp [Nat.succ_eq_add_one, Nat.right_distrib]
    simp [List.all_cons] at H
    simp [IH2 s H.2]
    rw [motive_1]
    apply H.1

end MLIR.AST.TensorElem


/-
### Vector types
TODO: Not modeled fully by lean-mlir right now
-/


/-
### Ranked tensors
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

-- Ranked tensors have known rank but possibly unknown dimensions. At runtime,
-- the size is fully determined. We store the visible type information in the
-- parameters of RankedTensor, and collect the fully-specified runtime data in
-- the structure fields.
--
-- We add an intermediate [size] parameter so that we can separate data
-- manipulation and proofs about the product of the dimensions, which would
-- otherwise be tricky due to [Fin] constantly carrying proofs.
structure RankedTensor (α: Type) (D: DimList) where
  -- Actual dimensions
  shape: List Nat
  -- Contents; we use a function for brevity
  -- TODO: RankedTensor: Consider a more computable data storage method
  data: List α
  -- Invariants: shape/dimension must be compatible, shape/size must match
  h_refines: shape_refines shape D
  h_data_size: data.length = shape_prod shape

theorem RankedTensor.eq_of_fields_eq (α D): ∀ (t₁ t₂: RankedTensor α D),
  t₁.shape = t₂.shape →
  t₁.data = t₂.data →
    t₁ = t₂ := by
  intros t₁ t₂ Hshape Hdata
  cases t₁; cases t₂; simp at *
  trivial

def RankedTensor.uniform {α} D (v: α): RankedTensor α D :=
  { shape       := D.default_refinement,
    data        := List.uniform v (shape_prod D.default_refinement),
    h_refines   := default_refinement_refines _,
    h_data_size := List.uniform_length _ _ }

def RankedTensor.default α D [Inhabited α]: RankedTensor α D :=
  RankedTensor.uniform D Inhabited.default

inductive MLIR.AST.TensorElem.rankCompatibleWith (D: DimList) (e:TensorElem) :=
  | Uniform i: e = TensorElem.int i → rankCompatibleWith D e
  | HasShape s: e.hasShape s → shape_refines s D → rankCompatibleWith D e

-- TODO: RankedTensor.ofTensorElem: account for typing?
def RankedTensor.ofTensorElem (D: DimList) (e: TensorElem)
    (H: e.rankCompatibleWith D): RankedTensor Int D:=
  match H with
  | TensorElem.rankCompatibleWith.Uniform i Heq =>
      RankedTensor.uniform D i
  | TensorElem.rankCompatibleWith.HasShape s Hshape Hrefines =>
      { shape       := s,
        data        := e.flatten,
        h_refines   := Hrefines,
        h_data_size := TensorElem.flatten_size e s Hshape }

instance {α D} [Inhabited α]: Inhabited (RankedTensor α D) where
  default := RankedTensor.default α D


/-
### Unranked tensors
TODO: Unranked tensors?
-/


/-
### Evaluation of MLIR types
TODO: Not all MLIRTy types are correctly evaluated
-/

/- MLIRTy is a nested inductive type, thus defined with well-founded recursion.
   This prevents it from being reduced by the elaborator, so instead we define
   it manually with the recursor.
   See: https://leanprover.zulipchat.com/#narrow/stream/270676-lean4/topic/
   reduction.20of.20dependent.20return.20type/near/276044057 -/

@[reducible, simp_itree]
def MLIR.AST.MLIRTy.eval (τ: MLIRTy): Type :=
  @MLIRTy.rec
    (motive_1 := fun _ => Type)
    (motive_2 := fun _ => Type)
    -- MLIRTy.fn
    (fun τ₁ τ₂ eval_τ₁ eval_τ₂ => eval_τ₁ → eval_τ₂)
    -- MLIRTy.int
    (fun bitsize => Int)
    -- MLIRTy.float
    (fun bitsize => Float)
    -- Mapping motive_2 to motive_1
    (fun _ ih => ih)
    -- MLIRTy.vector (todo)
    (fun D τ eval_τ => Unit)
    -- MLIRTy.tensor (todo)
    (fun D τ eval_τ => RankedTensor eval_τ D)
    -- MLIRTy.user (todo)
    (fun name => Unit)
    -- MLIRTy.tuple []
    Unit
    -- MLIRTy.tuple (τ::l)
    (fun τ l eval_τ eval_l =>
      match l with
      | [] => eval_τ
      | _  => eval_τ × eval_l)
    τ

@[reducible]
def MLIR.AST.MLIRTy.evalList (l: List MLIRTy) : Type :=
  @MLIRTy.rec_1
    -- Same as above
    (motive_1 := fun _ => Type)
    (motive_2 := fun _ => Type)
    (fun τ₁ τ₂ eval_τ₁ eval_τ₂ => eval_τ₁ → eval_τ₂)
    (fun bitsize => Int)
    (fun bitsize => Float)
    (fun _ ih => ih)
    (fun D τ eval_τ => Unit)
    (fun D τ eval_τ => RankedTensor eval_τ D)
    (fun name => Unit)
    Unit
    (fun τ l eval_τ eval_l =>
      match l with
      | [] => eval_τ
      | _  => eval_τ × eval_l)
    l

mutual
  def MLIR.AST.MLIRTy.default (τ: MLIRTy): τ.eval :=
    match τ with
    | MLIRTy.fn τ₁ τ₂ => (fun _ => τ₂.default)
    | MLIRTy.int _ => (0:Int)
    | MLIRTy.float _ => (0.0:Float)
    | MLIRTy.tuple l => MLIRTy.defaultList l
    | MLIRTy.vector _ _ => () /- todo -/
    | MLIRTy.tensor D τ => @RankedTensor.default τ.eval D ⟨default τ⟩
    | MLIRTy.user _ => () /- todo -/

  protected def MLIR.AST.MLIRTy.defaultList (l: List MLIRTy):
      MLIRTy.evalList l :=
    match l with
    | [] => ()
    | [τ] => τ.default
    | τ₁ :: τ₂ :: l => (τ₁.default, MLIRTy.defaultList (τ₂::l))
end

instance (τ: MLIRTy): Inhabited τ.eval where
  default := τ.default
