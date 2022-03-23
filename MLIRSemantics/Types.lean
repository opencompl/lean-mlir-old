/-
## MLIR types

This file implements support for builtin MLIR types, as well as conversion to
concrete Lean types and values.

In order to support the untyped SSA environment (which doesn't require name
definitions/uses to match as this is proven later), every concrete MLIR type
should be Inhabited so that `SSAEnvE.Get` can return default values.

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

import MLIR.AST
open MLIR.AST


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
def shape_prod :=
  List.foldr (·*·) 1

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
    simp [List.foldr]

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
  size: Nat
  data: Fin size → α
  -- Invariants: shape/dimension must be compatible, shape/size must match
  Hdim: shape_refines shape D
  Hsize: size = shape_prod shape

theorem RankedTensor.eq_of_fields_eq (α D): ∀ (t₁ t₂: RankedTensor α D),
  t₁.shape = t₂.shape →
  (Hsize: t₁.size = t₂.size) →
  (Hdata: HEq t₁.data t₂.data) →
    t₁ = t₂ := by
  intros t₁ t₂ Hshape Hsize Hdata
  cases t₁; cases t₂; simp at *
  trivial

def RankedTensor.default {α D} [Inhabited α]: RankedTensor α D :=
  { shape := D.default_refinement,
    size  := shape_prod (D.default_refinement),
    data  := fun _ => Inhabited.default,
    Hdim  := default_refinement_refines _,
    Hsize := rfl }

instance {α D} [Inhabited α]: Inhabited (RankedTensor α D) where
  default := RankedTensor.default

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

@[reducible]
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
