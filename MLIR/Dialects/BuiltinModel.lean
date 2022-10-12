/-
# Data model for the built-in dialect
-/

import MLIR.AST
import MLIR.Doc
import MLIR.Semantics.Types
import MLIR.Semantics.TensorElem
import MLIR.Util.Mathlib4.NatBasic
import MLIR.Util.Mathlib4.Dvd
import MLIR.Util.Mathlib4.NatLemmas
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
  h_data_size: data.length = shapeProd shape


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
    data        := List.uniform v (shapeProd shape),
    h_data_size := List.length_uniform _ _ }

instance {τ}: Inhabited (Tensor τ) where
  default := Tensor.uniform [1] default


-- Map a function over a tensor.
def Tensor.map {σ τ} (v: Tensor σ) (f: σ.eval → τ.eval): Tensor τ :=
 Tensor.mk (shape := v.shape) (data := v.data.map f) (h_data_size := by {
    rewrite [List.length_map]
    apply v.h_data_size;
 })

#check Tensor.map

theorem Tensor.map_shape {σ τ: MLIRTy} (v: Tensor σ) (f: σ.eval → τ.eval): v.shape = (Tensor.map v f).shape := by {
  simp [Tensor.map];
}


instance {τ}: DecidableEq (Tensor τ) := fun t₁ t₂ => by
  cases t₁; cases t₂; simp
  exact inferInstance


def Tensor.mapWithFlatIndex {σ τ} (v: Tensor σ) (f: TensorFlatIndex (shapeProd v.shape) → σ.eval → τ.eval): Tensor τ :=
  Tensor.mk (shape := v.shape)
    (data := (List.zipFlatIndex v.data).map (fun (val, ix) => f (v.h_data_size ▸ ix) val)) (h_data_size := by simp; apply v.h_data_size)


-- getter at a flat index
def Tensor.getAtFlatIndex {σ} (v: Tensor σ) (ix: TensorFlatIndex (shapeProd v.shape)): σ.eval :=
  List.getF v.data ix.ix (h := by {  rewrite [v.h_data_size]; exact ix.h_ix_inbound; })


theorem arg_equal_implies_fn_equal (α β: Type) (x y :α) (f: α -> β) (EQ: x = y): f x = f y := by {
  simp [EQ];
}

theorem arg_equal_implies_fn_equal_2 (α β γ: Type) (a a' :α) (b b': β) (f: α -> β -> γ)
   (EQα: a = a') (EQβ: b = b') : f a b = f a' b' := by {
  simp [EQα, EQβ];
}


-- getting at a mapped flat index returns the value that's computed by running the map fn at
-- that index.
def Tensor.mapWithFlatIndexCorrect
   {σ τ: MLIRTy} (v: Tensor σ) (f: TensorFlatIndex (shapeProd v.shape) → σ.eval → τ.eval)
   (ix: TensorFlatIndex (shapeProd v.shape)):
  (v.mapWithFlatIndex f).getAtFlatIndex ix = f ix (v.getAtFlatIndex ix) := by {
  simp [Tensor.getAtFlatIndex];
  simp [Tensor.mapWithFlatIndex];
  rewrite [List.zip_flat_index_get (xs := v.data) (getIx := ix.ix)
      (GETIX := by {  rewrite [v.h_data_size];  simp[ix.h_ix_inbound]; } )];
  simp;
  rewrite [TensorFlatIndex.cast_left];
  rfl;
}



/-
## Ranked tensor type

A ranked tensor extends the runtime tensor with a statically-known list of
dimensions (all of which may not be known) which adds more invariants.
-/

structure RankedTensor (D: DimList) (τ: MLIRTy) extends Tensor τ where
  -- Invariants: shape/dimension must be compatible, shape/size must match
  h_refines: D.shapeRefines shape

theorem RankedTensor.eq_of_fields_eq {τ D} (t₁ t₂: RankedTensor τ D):
    t₁.shape = t₂.shape → t₁.data = t₂.data → t₁ = t₂ := by
  intros h_shape h_data
  suffices t₁.toTensor = t₂.toTensor by
    cases t₁; cases t₂; simp at *; trivial
  apply Tensor.eq_of_fields_eq <;> trivial

instance {τ D}: DecidableEq (RankedTensor τ D) := fun t₁ t₂ => by
  cases t₁; cases t₂; simp
  exact inferInstance

def RankedTensor.uniform {τ} (D: DimList) (v: τ.eval): RankedTensor D τ :=
  { Tensor.uniform D.defaultRefinement v with
    h_refines   := DimList.defaultRefinement_refines _ }

instance {τ D}: Inhabited (RankedTensor D τ) where
  default := RankedTensor.uniform D default

def RankedTensor.typeStr (τ: MLIRTy) (D: DimList): String :=
  let dims := "x".intercalate (D.map (MLIR.Doc.Pretty.doc ·))
  s!"tensor<{dims}x{τ}>"

def RankedTensor.str {τ D} (t: RankedTensor D τ): String :=
  let dims := "×".intercalate (D.map (MLIR.Doc.Pretty.doc ·))
  let data := "[" ++ " ".intercalate (t.data.map toString) ++ "]"
  s!"({dims}){data}"

-- Map a function over a tensor.
def RankedTensor.map {σ σ': MLIRTy} {D: DimList} (v: RankedTensor D σ) (f: σ.eval → σ'.eval): RankedTensor D σ' :=
  let t' : Tensor σ' := v.toTensor.map f
  let H : D.shapeRefines t'.shape := by {
      rewrite [← Tensor.map_shape];
      simp [v.h_refines];
  };
  RankedTensor.mk t' H

-- Conversion from TensorElem

def RankedTensor.ofTensorLiteral (lit: TensorLiteral D τ): RankedTensor D τ :=
  match τ, lit, lit.h_rank with
  | .int sgn 1, lit, .UniformBool b _ _ =>
      RankedTensor.uniform D (FinInt.ofInt 1 (if b then 1 else 0))
  | .int sgn sz, lit, .UniformInt i _ _ _ _ =>
      RankedTensor.uniform D (FinInt.ofInt sz i)
  | .float bitsize, lit, .UniformFloat f _ _ =>
      RankedTensor.uniform D f
  | τ, lit, .HasShape s _ Hshape Hrefines =>
      { shape       := s,
        data        := lit.elem.flatten lit.h_type,
        h_refines   := Hrefines,
        h_data_size := TensorElem.flatten_size lit.elem s Hshape lit.h_type }

def RankedTensor.ofTensorElem {τ} (D: DimList) (elem: TensorElem)
    (h_type: elem.hasType τ) (h_rank: elem.rankCompatibleWith D τ):
    RankedTensor D τ :=
  ofTensorLiteral { elem, h_type, h_rank }


open DimList in
def reshape {τ} {D: DimList} (D': DimList)
    (H: D.known) (H': D'.known) (Hprod: D'.prod = D.prod):
    RankedTensor D τ → RankedTensor D' τ :=
  fun t =>
    { shape       := D'.project,
      data        := t.data
      h_refines   := dim_known_project_refines H',
      h_data_size := by rw [t.h_data_size, dim_known_prod D' H', Hprod]
                        rw [dim_known_prod_refines H]
                        apply t.h_refines }

theorem reshape_reshape {τ} {D: DimList} (D₁ D₂: DimList)
    (H: D.known) (H₁: D₁.known) (H₂: D₂.known)
    (Hprod₁: D₁.prod = D.prod) (Hprod₂: D₂.prod = D₁.prod)
    (t: RankedTensor D τ):
      reshape D₂ H₁ H₂ Hprod₂ (reshape D₁ H H₁ Hprod₁ t) =
      reshape D₂ H H₂ (Eq.trans Hprod₂ Hprod₁) t :=
  rfl

open DimList in
theorem reshape_self {τ} D H₁ H₂ Hprod (t: RankedTensor D τ):
    reshape D H₁ H₂ Hprod t = t := by
  simp [reshape, dim_known_project_eq H₁ t.h_refines]


-- Type interface for registration with MLIRType

private abbrev σ_RankedTensor := DimList × MLIRTy
private abbrev ε_RankedTensor := fun (D, τ) => RankedTensor D τ

instance: DialectTypeIntf σ_RankedTensor ε_RankedTensor where
  inhabited := default
  typeEq := inferInstance
  eq := inferInstance
  str := fun (τ, D) t => t.str
  typeStr := fun (τ, D) => RankedTensor.typeStr D τ

/-
## Unranked tensor type
-/

abbrev UnrankedTensor := Tensor

-- Type interface for registration with MLIRType

private abbrev σ_UnrankedTensor := MLIRTy
private abbrev ε_UnrankedTensor := UnrankedTensor

def UnrankedTensor.typeStr (τ: MLIRTy): String :=
  s!"tensor<*{τ}>"

-- TODO: String representation of ranked tensors
def UnrankedTensor.str {τ} (t: UnrankedTensor τ) :=
  s!"<an unranked tensor of base type {τ}>"

instance: DialectTypeIntf σ_UnrankedTensor ε_UnrankedTensor where
  inhabited := default
  typeEq := inferInstance
  eq := inferInstance
  str := @UnrankedTensor.str
  typeStr := UnrankedTensor.typeStr


/-
## Vector type
-/

def List.zipMul (as: List Nat) (bs: List Nat) (H: as.length = bs.length): List Nat :=
 match as with
 | [] => match bs with
         | [] => []
         | b::bs' => nomatch H
 | a::as' =>
    match bs with
    | [] => nomatch H
    | b::bs' => (a*b):: List.zipMul as' bs' (by { simp at H; assumption })


def Vector.size (fixed scalable: List Nat) (scale: List Nat)
    (H: scale.length = scalable.length) :=
  shapeProd fixed * shapeProd (List.zipMul scale scalable H)

structure Vector (fixed: List Nat) (scalable: List Nat) (τ: MLIRTy) where
  -- Scales (number of instantiations of each scalable dimension)
  scale: List Nat
  -- Contents in row-major order
  data: List τ.eval
  -- Invariant: Number of scales must match number of scalable dimensions
  h_scale_size: scale.length = scalable.length
  -- Invariant: Number of data elements must match scale
  h_data_size: data.length = Vector.size fixed scalable scale h_scale_size

theorem Vector.eq_of_fields_eq (v₁ v₂: Vector τ fixed scalable):
    v₁.scale = v₂.scale → v₁.data = v₂.data → v₁ = v₂ := by
  intros h_scale h_data
  cases v₁; cases v₂; simp at *
  trivial

instance: Inhabited (Vector fixed scalable τ) where
  default :=
    { scale := List.map (fun _ => 1) scalable,
      data := List.uniform default (Vector.size fixed scalable
                (List.map (fun _ => 1) scalable) (by apply List.length_map)),
      h_scale_size := by apply List.length_map,
      h_data_size := by apply List.length_uniform }

instance: DecidableEq (Vector τ fixed scalable) := fun v₁ v₂ =>
  if h: v₁.scale = v₂.scale ∧ v₁.data = v₂.data then
    isTrue (by apply Vector.eq_of_fields_eq _ _ h.1 h.2)
  else
    isFalse fun h' => by simp [h'] at *

def Vector.typeStr (fixed scalable: List Nat) (τ: MLIRTy) :=
  let sf := "x".intercalate (fixed.map (MLIR.Doc.Pretty.doc ·))
  let ss := "x".intercalate (scalable.map (MLIR.Doc.Pretty.doc ·))

  if fixed.length > 0 && scalable.length > 0 then
    s!"vector<{sf}x[{ss}]x{τ}>"
  else if fixed.length > 0 then
    s!"vector<{sf}x{τ}>"
  else if scalable.length > 0 then
    s!"vector<[{ss}]x{τ}>"
  else
    s!"vector<{τ}>"

-- TODO: String representation of vector values
def Vector.str (v: Vector fixed scalable τ): String :=
  let str_type := Vector.typeStr fixed scalable τ
  s!"<a vector of type {str_type}>"

-- Type interface for registration with MLIRType

private abbrev σ_Vector := List Nat × List Nat × MLIRTy
private abbrev ε_Vector := fun (fixed, scalable, τ) => Vector fixed scalable τ

instance: DialectTypeIntf σ_Vector ε_Vector where
  inhabited := default
  typeEq := inferInstance
  eq := inferInstance
  str := fun (τ, fixed, scalable) t => t.str
  typeStr := fun (τ, fixed, scalable) => Vector.typeStr τ fixed scalable


/-
## TODO: Memref types
-/

inductive MemrefLayoutSpec where
| stride: (offset: Dimension) -> (stride: List Dimension) -> MemrefLayoutSpec
| attr: AttrVal -> MemrefLayoutSpec

partial def docMemrefLayoutSpec(spec: MemrefLayoutSpec) : MLIR.Doc.Doc :=
match spec with
| MemrefLayoutSpec.stride offset strides => [doc| "offset:" offset ", strides: " "[" (strides),* "]"]
| MemrefLayoutSpec.attr v => docAttrVal v

-- Text representation

/-| MLIRTy.memrefRanked dims ty layout? memspace? =>
    let docLayout := match layout? with | some x => [doc| "," (docMemrefLayoutSpec x)] | none => ""
    let docMemspace := match memspace? with | some x => [doc| "," (docAttrVal x)] | none => ""
    [doc| "memref<" (intercalate_doc dims "x") "x" (go ty) (docLayout)  (docMemspace) ">"]
  | MLIRTy.memrefUnranked ty memspace? =>
    let docMemspace := match memspace? with | some x => [doc| "," (docAttrVal x)] | none => ""
    [doc| "memref<" "*x" (go ty) (docMemspace) ">"] -/


/-
## Summary of types
-/

abbrev builtin.σ :=
  (σ_RankedTensor ⊕ σ_UnrankedTensor) ⊕ σ_Vector
abbrev builtin.ε :=
  Sum.cases
    (Sum.cases ε_RankedTensor ε_UnrankedTensor)
    ε_Vector

@[matchPattern]
def builtin.σ.tensor (D: DimList) (τ: MLIRTy): builtin.σ :=
  Sum.inl (Sum.inl (D, τ))

@[matchPattern]
def builtin.σ.tensor_unranked (τ: MLIRTy): builtin.σ :=
  Sum.inl (Sum.inr τ)

@[matchPattern]
def builtin.σ.vector (fixed scalable: List Nat) (τ: MLIRTy): builtin.σ :=
  Sum.inr (fixed, scalable, τ)


/-
## Dense vector/tensor attribute
-/

structure DenseAttr: Type where
  elem: TensorElem
  τ_sig: builtin.σ
deriving DecidableEq

-- TODO: String representation of dense<> attributes via TensorElem
def DenseAttr.str (a: DenseAttr): String :=
  let τ_str :=
    match a.τ_sig with
    | builtin.σ.tensor τ D => RankedTensor.typeStr D τ
    | builtin.σ.tensor_unranked τ => UnrankedTensor.typeStr τ
    | builtin.σ.vector f s τ => Vector.typeStr f s τ
  s!"dense<...>: {τ_str}"

private abbrev α_DenseAttr := DenseAttr

instance: DialectAttrIntf α_DenseAttr where
  eq := inferInstance
  str := DenseAttr.str


/-
## Summary of attributes
-/

abbrev builtin.α :=
  α_DenseAttr


/-
## Builtin dialect definition
-/

instance builtin: Dialect builtin.α builtin.σ builtin.ε where
  name := "builtin"
  iα := inferInstance
  iε := inferInstance

-- Custom types

@[matchPattern, simp]
def builtin.tensor (D: DimList) (τ: MLIRTy): MLIRType builtin :=
  MLIRType.extended (builtin.σ.tensor D τ)

@[matchPattern]
def builtin.tensor_unranked (τ: MLIRTy): MLIRType builtin :=
  MLIRType.extended (builtin.σ.tensor_unranked τ)

@[matchPattern]
def builtin.vector (fixed scalable: List Nat) (τ: MLIRTy):
    MLIRType builtin :=
  MLIRType.extended (builtin.σ.vector fixed scalable τ)

@[matchPattern]
def builtin.memref (D: DimList) (τ: MLIRTy) (layout: Option MemrefLayoutSpec)
    (memspace: Option AttrVal): MLIRType builtin :=
  MLIRType.undefined "builtin.memref"

@[matchPattern]
def builtin.memref_unranked (τ: MLIRTy) (memspace: Option AttrVal):
    MLIRType builtin :=
  MLIRType.undefined "builtin.memref_unranked"

-- Custom attributes

@[matchPattern]
def builtin.dense_vector_attr (e: TensorElem) (fixed scalable: List Nat)
    (τ: MLIRTy): AttrValue builtin :=
  AttrValue.extended (DenseAttr.mk e (builtin.σ.vector fixed scalable τ))

@[matchPattern]
def builtin.dense_tensor_attr (e: TensorElem) (D: DimList) (τ: MLIRTy):
    AttrValue builtin :=
  AttrValue.extended (DenseAttr.mk e (builtin.σ.tensor D τ))

@[matchPattern]
def builtin.dense_attr (e: TensorElem) {s: builtin.σ}: AttrValue builtin :=
  AttrValue.extended (DenseAttr.mk e s)


/-
## High-level utilities
-/

-- Create a dense vector from a vector type
-- FIXME: Does AttrVal.dense actually also support tensor types??
def builtin.denseWithType (e: TensorElem) (τ: MLIRType builtin):
    AttrValue builtin :=
  match τ with
  | builtin.tensor D τ =>
      builtin.dense_tensor_attr e D τ
  | builtin.vector fixed scalable τ =>
      builtin.dense_vector_attr e fixed scalable τ
  | _ =>
      panic! s!"buitin.denseVectorWithType: {τ} not a vector type"

-- Create a dense vector with values `xs` and type `vector<len(xs)*ity>`
def builtin.denseVectorOfList (xs: List Int) (ity: MLIRTy := .i32):
    AttrValue builtin :=
  builtin.dense_vector_attr xs [xs.length] [] ity

