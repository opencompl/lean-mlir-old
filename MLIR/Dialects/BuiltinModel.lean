/-
## Data model for the built-in dialect
-/

import MLIR.AST
import MLIR.Doc
import MLIR.Semantics.Types
open MLIR.AST

/-
## General tensor type

The following structure is used as the basis for both ranked and unranked
tensor types, since both have the same runtime contents and only differ by the
amount of information exposed in the typing system.

At runtime the dimension of tensors is always known. As a general rule, we
store compile-time type information in the parameters of the structure and the
runtime contents in the fields.
-/

structure Tensor (τ: MLIRTy) where
  -- Runtime-determined dimensions
  shape: List Nat
  -- Contents in row-major order
  data: List τ.eval
  -- Invariant: shape and size must match
  h_data_size: data.length = shape_prod shape

theorem Tensor.eq_of_fields_eq {τ} (t₁ t₂: Tensor τ):
    t₁.shape = t₂.shape → t₁.data = t₂.data → t₁ = t₂ := by
  intros h_shape h_data
  cases t₁; cases t₂; simp at *
  trivial

instance {τ}: DecidableEq (Tensor τ) := fun t₁ t₂ => by
  cases t₁; cases t₂; simp
  exact inferInstance

def Tensor.uniform {τ} (shape: List Nat) (v: τ.eval): Tensor τ :=
  { shape       := shape,
    data        := List.uniform v (shape_prod shape),
    h_data_size := List.uniform_length _ _ }

instance {τ}: Inhabited (Tensor τ) where
  default := Tensor.uniform [1] default


/-
## Ranked tensor type

A ranked tensor extends the runtime tensor with a statically-known list of
dimensions (all of which may not be known) which adds more invariants.
-/

structure RankedTensor (τ: MLIRTy) (D: DimList) extends Tensor τ where
  -- Invariants: shape/dimension must be compatible, shape/size must match
  h_refines: shape_refines shape D

theorem RankedTensor.eq_of_fields_eq {τ D} (t₁ t₂: RankedTensor τ D):
    t₁.shape = t₂.shape → t₁.data = t₂.data → t₁ = t₂ := by
  intros h_shape h_data
  suffices t₁.toTensor = t₂.toTensor by
    cases t₁; cases t₂; simp at *; trivial
  apply Tensor.eq_of_fields_eq <;> trivial

instance {τ D}: DecidableEq (RankedTensor τ D) := fun t₁ t₂ => by
  cases t₁; cases t₂; simp
  exact inferInstance

def RankedTensor.uniform {τ} (D: DimList) (v: τ.eval): RankedTensor τ D :=
  { Tensor.uniform D.default_refinement v with
    h_refines   := default_refinement_refines _ }

instance {τ D}: Inhabited (RankedTensor τ D) where
  default := RankedTensor.uniform D default

-- Conversion from TensorElem

def RankedTensor.ofTensorElem {τ} (D: DimList) (e: TensorElem)
    (Htype: e.hasType τ) (Hcompat: e.rankCompatibleWith D τ):
    RankedTensor τ D:=
  match Hcompat with
  | .UniformInt i bitsize _ =>
      RankedTensor.uniform D i
  | .UniformBool b _ =>
      RankedTensor.uniform D (if b then 1 else 0)
  | .UniformFloat f bitsize _ =>
      RankedTensor.uniform D f
  | .HasShape s τ Hshape Hrefines =>
      { shape       := s,
        data        := e.flatten Htype,
        h_refines   := Hrefines,
        h_data_size := TensorElem.flatten_size e s Hshape Htype }

-- Type interface for registration with MLIRTy
-- TODO: String representation of tensors (same for unranked tensors)

instance MLIRTy.rankedTensorType {τ D}: TypeIntf (RankedTensor τ D) where
  eq := inferInstance
  inhabited := inferInstance
  str := ⟨fun t =>
    let dims := "×".intercalate (D.map (MLIR.Doc.Pretty.doc ·))
    s!"<a tensor of size {dims} and base type {τ}>"⟩

instance MLIRTy.builtin.tensor:
    TypeFamilyIntf "builtin.tensor" (MLIRTy × DimList) where
  α := fun (τ, D) => RankedTensor τ D
  compare := inferInstance
  str := ⟨fun (τ, D) =>
    let dims := "x".intercalate (D.map (MLIR.Doc.Pretty.doc ·))
    s!"tensor<{dims}x{τ}>"⟩
  eval := fun (τ, D) => inferInstance

def MLIRTy.tensorRanked τ D :=
  @MLIRTy.generic "builtin.tensor" _ (τ, D) MLIRTy.builtin.tensor


/-
## Unranked tensor type
-/

abbrev UnrankedTensor := Tensor

-- Type interface for registration with MLIRTy

instance MLIRTy.unrankedTensorType {τ}: TypeIntf (UnrankedTensor τ) where
  eq := inferInstance
  inhabited := inferInstance
  str := ⟨fun t => s!"<an unranked tensor of base type {τ}>"⟩

-- TODO: Ranked and unranked tensors do not share the same name
instance MLIRTy.builtin.unranked_tensor:
    TypeFamilyIntf "builtin.unranked_tensor" MLIRTy where
  α := UnrankedTensor
  compare := inferInstance
  str := ⟨fun τ => s!"tensor<*{τ}>"⟩
  eval := fun τ => inferInstance

def MLIRTy.tensorUnranked τ :=
  @MLIRTy.generic "builtin.unranked_tensor" _ τ MLIRTy.builtin.unranked_tensor


/-
## TODO: Vector and memref types
-/

def MLIRTy.vector (fixed: List Int) (scaled: List Int) (τ: MLIRTy) :=
  MLIRTy.undefined "builtin.vector"

def MLIRTy.memrefRanked (D: DimList) (τ: MLIRTy)
    (layout: Option MemrefLayoutSpec) (memspace: Option AttrVal) :=
  MLIRTy.undefined "builtin.memref"

def MLIRTy.memrefUnranked (τ: MLIRTy) (memspace: Option AttrVal) :=
  MLIRTy.undefined "builtin.memref_unranked"


/-
## TODO: Non-trivial builtin attributes
-/

-- | create a dense vector with values 'xs' and type vector<len(xs)xity>
-- FIXME: AttrVal.dense_vector
def AttrVal.dense_vector (xs: List Int) (ity: MLIRTy := MLIRTy.int 32): AttrVal :=
  let fixedShape := [Int.ofNat xs.length]
  let scaledShape := []
  let vty := MLIRTy.vector fixedShape scaledShape ity 
  AttrVal.dense xs vty
