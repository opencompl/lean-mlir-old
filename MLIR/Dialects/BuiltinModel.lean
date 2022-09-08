/-
# Data model for the built-in dialect
-/

import MLIR.AST
import MLIR.Doc
import MLIR.Semantics.Types
import MLIR.Semantics.TensorElem
open MLIR.AST
open MLIR.AST.TensorElem (shapeProd)

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


/-
A 1D index into a tensor. Witnesses that the flat index is in bounds of the shape of the tensor.
-/
structure TensorFlatIndex (bound: Nat) where
  ix: Nat
  h_ix_inbound: (ix < bound)

-- shape: [S1, S2, S3]:
-- indexes: [A1, A2, A3]:
-- flattened index:
--    A1 + S1 (A2 + S2(A3))
structure TensorIndex (shape: List Nat) where
  ixs: List Nat -- indexes into the tensor
  h_ix_length: ixs.length = shape.length -- Invariant: we have as many indexes as there are dimensions
  h_ix_bound: ∀ (i: Nat) (INBOUNDS: i < shape.length),
    List.getF ixs i (h_ix_length ▸ INBOUNDS) < shape[i] -- Invariant: the dimensions are inbounds.


/-
Projecting out innermost dimension.
-/
open Lean in
def TensorIndex.projectOut {outermost: Nat} {shape: List Nat} (index: TensorIndex (outermost :: shape)): TensorIndex shape :=
  match H:index.ixs with
  | [] => by {
    have CONTRA := index.h_ix_length;
    simp [H] at CONTRA;
    }
  | ix :: ixs' => {
     ixs := ixs'
     h_ix_length := by {
       have LEN := index.h_ix_length;
       rewrite [H] at LEN;
       simp [List.length] at LEN;
       exact LEN;
     }
     h_ix_bound := by {
       have BOUND := index.h_ix_bound;
       intros i;
       intros I_INBOUND;
       specialize (BOUND (i + 1));
       simp[H, List.getF] at BOUND;
       apply BOUND;
       simp [List.length];
       have I_INBOUND' := Nat.add_lt_add_right (k := 1) I_INBOUND;
       apply I_INBOUND';

     }
  }

-- shape: [S1, S2, S3]:
-- indexes: [A1, A2, A3]:
-- flattened index:
--    A1 + S1 (A2 + S2(A3))
def TensorIndex.linearize {innerDim: Nat} {restDims: List Nat} (index: TensorIndex (innerDim :: restDims)): Nat :=
  let IX0: 0 < index.ixs.length := by {
    rewrite [index.h_ix_length];
    simp;
    apply Nat.zero_lt_of_ne_zero;
    simp;
  }
  let ix0 := index.ixs[0]'IX0
  match restDims with
  | [] => ix0
  | _outermost ::_restDims  => ix0 + innerDim * (TensorIndex.linearize index.projectOut)


-- #check Nat.mod_lt
-- Delinearlize the outermost dimension of size 'size' into 'modulus * (size/modulus)'
def TensorIndex.delinearizeInnermost {innerDim: Nat} {restDims: List Nat}
  (modulus: Nat)
  (MOD_GT_0: modulus > 0)
  (index: TensorIndex (innerDim :: restDims)): TensorIndex (modulus :: (innerDim/modulus) :: restDims) :=
  match H:index.ixs with
  | [] => by {
    have CONTRA := index.h_ix_length;
    rewrite [H] at CONTRA;
    simp at CONTRA;
  }
  | innerIx :: ixs' =>
    let ix0 := innerIx % modulus;
    let ix1 := innerIx / modulus;
    TensorIndex.mk
      (ixs := ix0 :: ix1 :: ixs')
      (h_ix_length := by {
         have HLEN := index.h_ix_length;
         rewrite [H] at HLEN;
         simp at HLEN;
         simp [index.h_ix_length, HLEN];
      }) (h_ix_bound := fun i I_INBOUND =>
           match IVAL: i with
           | 0 => by {
             simp [List.getF];
             apply Nat.mod_lt;
             apply MOD_GT_0;
           }
           | 1 => by {
             simp [List.getF];
             -- ⊢ innerIx / modulus < innerDim / modulus
             have INNERIX : innerIx < innerDim := by {
                 have h := index.h_ix_bound;
                 specialize h (i := 0);
                 specialize h (by {
                  simp; apply Nat.zero_lt_of_ne_zero; simp;
                 });
                 simp [H, List.getF] at h;
                 apply h;
              }
             sorry_arith -- innerIx < innerDim => (innerIx / modulus) < (innerDim / modulus)
           }
           | Nat.succ (Nat.succ i') => by {
                simp [H, List.getF];
                have h := index.h_ix_bound;
                specialize h (i' + 1);
                simp [H, List.length, List.getF, List.length] at h;
                apply h;
                simp [Nat.add_one];
                simp [List.length] at I_INBOUND;
                -- I_INBOUND : Nat.succ (Nat.succ i') < List.length restDims + 1 + 1
                -- ⊢ i' + 1 ≤ List.length restDims +1
                apply Nat.lt_of_succ_lt_succ;
                apply I_INBOUND;
           }
      )


-- Delinearization is correct iff the index expression of the lineraized
-- case is equal to the index expression after delin.
theorem TensorIndex.delineraize_correct
  {innerDim: Nat} {restDims: List Nat}
  (index: TensorIndex (innerDim :: restDims))
  (modulus: Nat)
  (MOD_GT_0: modulus > 0): (index.delinearizeInnermost modulus MOD_GT_0).linearize = index.linearize := by {
  sorry -- post-dinner proof.
}


/-
def TensorIndex.ofFlatIndexGo (rest: Nat) (shape: List Nat)
  (rest_inbound: rest < shapeProd shape)
  (index: TensorIndex shape): TensorIndex shape :=
  match shape with
  | [] => index
  | s :: shapes =>
    let ix := rest % s
    let rest' := rest / s
    TensorIndex.mk
      (ixs := ix::index.ixs)
      (h_ix_length := by {
        simp;
        rewrite [index.h_ix_length];
      })
      (h_ix_bound := sorry)
-/

theorem shapeProd_cons_prod (x y: Nat) (zs: List Nat): shapeProd (x :: y :: zs) = shapeProd ((x *y) :: zs) := by {
   simp [shapeProd, List.foldr];
   simp;
   sorry_arith;
}


-- shape: 2x3x5:
-- 0:(0,0,0)
-- 0:(0,0,0)
-- 0:(0,0,0)
-- 0:(0,0,0)
-- 0:(0,0,0)
-- 0:(0,0,0)
-- 0:(0,0,0)
-- | TODO: fix sorrys about hypotheses.
def TensorIndex.ofFlatIndex {innerDim: Nat} {restDims: List Nat}
  (INNERDIM: innerDim > 0)
  (flat: TensorFlatIndex (shapeProd (innerDim :: restDims))): TensorIndex (innerDim :: restDims) :=
   -- | extract out code to create tensor index from tensor flat index.
   match restDims with
   | [] => TensorIndex.mk (ixs := [flat.ix]) (by simp) (by {
        intros i I_INBOUND;
        simp [List.length] at I_INBOUND;
        have I_EQ_0 : i = 0 :=
          match i with
          | 0 => by simp;
          | Nat.succ i' => by {
               simp at I_INBOUND;
               exact (nomatch I_INBOUND);
          };
       simp [I_EQ_0, List.getF];
       have IX_INBOUND := flat.h_ix_inbound;
       simp [shapeProd, List.foldr] at IX_INBOUND;
       apply IX_INBOUND;
     })
   | restDim0 :: restDims' =>
       let twoFlat : TensorIndex (innerDim * restDim0 :: restDims') :=
         TensorIndex.ofFlatIndex
              (sorry)
              (shapeProd_cons_prod innerDim restDim0 restDims' ▸ flat)
              (innerDim := innerDim*restDim0)
              (restDims := restDims');
       have RESTDIM0 : innerDim * restDim0 / innerDim = restDim0 := by {
            rewrite [Nat.mul_comm];
            apply Nat.mul_div_cancel;
            exact INNERDIM;
       }
       let final : TensorIndex (innerDim :: restDim0 :: restDims') :=
            RESTDIM0 ▸ TensorIndex.delinearizeInnermost  (modulus := innerDim) (sorry) twoFlat
       final




/-
def to_flat_index_go {τ: MLIRTy}
  (strides: List Nat)
  (ixs: List Nat)
  (h_ix_length: ixs.length = strides.length): Nat :=
  match strides with
  n| [] => 0
  | stride::strides' =>
    match ixs with
    | [] => by {
        simp at h_ix_length;
      }
    | ix::ixs' =>
        let H' : ixs'.length = strides'.length := by {
          simp at h_ix_length;
          exact h_ix_length;
        }
        ix + stride * (to_flat_index_go (τ := τ) strides' ixs' H')


def TensorIndex.ofFlat (f: TensorIndexFlat τ)
-/

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

theorem Tensor.map_shape {σ τ: MLIRTy} (v: Tensor σ) (f: σ.eval → τ.eval): v.shape = (v.map f).shape := by {
  simp [Tensor.map];
}

-- Helper function to zip a list with the index of the current value
def zipFlatIndexGo (xs: List α) (ix: Nat) (bound: Nat) (H: ix + xs.length = bound): List (α × TensorFlatIndex bound) :=
  match xs with
  | [] => []
  | x::xs' =>
     let ix_inbounds : ix < bound := by {
      rewrite [← H];
      apply Nat.lt_add_of_pos_right;
      simp;
      apply Nat.zero_lt_succ;
     }
     let ix' := ix + 1
     let H' :ix' + xs'.length = bound := by {
       rewrite [← H];
       simp;
       rewrite [Nat.succ_eq_add_one];
       -- ⊢ ix + 1 + List.length xs' = ix + (List.length xs' + 1)
       sorry_arith;
     }
     (x, TensorFlatIndex.mk ix ix_inbounds) :: zipFlatIndexGo xs' ix' bound H'


-- zipFlatIndexGo maintains length of the list.
theorem zip_flat_index_go_length (xs: List α): ∀ (ix: Nat) (bound: Nat) (H: ix + xs.length = bound),
  xs.length = (zipFlatIndexGo xs ix bound H).length := by {
  induction xs;
  case nil => {
    intros; unfold zipFlatIndexGo; rfl;
  }
  case cons x xs' IND => {
    intros ix bound H;
    simp [zipFlatIndexGo];
    apply IND;
  }
}
#check Nat.zero_lt_of_lt

-- Getting an element from the flattened zip
theorem List.zip_flat_index_go_get (xs: List α) (ix: Nat) (bound: Nat) (H: ix + xs.length = bound)
  (deltaIx: Nat) (GETIX: ix + deltaIx < xs.length):
  ((zipFlatIndexGo xs ix bound H).getF (ix + deltaIx) (zip_flat_index_go_length xs ix bound H ▸ GETIX)).snd.ix = ix + deltaIx := by {
  sorry
  /-
  induction xs;
  case nil => {
      simp [List.length, Nat.not_lt_zero] at GETIX;
  }
  case cons x xs' IND => {
    induction deltaIx;
    case zero => {
     simp [zipFlatIndexGo]; simp [List.getF];
    }
    simp[zipFlatIndexGo];
    simp;
  }
  sorry
 -/
}

-- Zip a list with the index of the current value
def List.zipFlatIndex (xs: List α): List (α × TensorFlatIndex xs.length) :=
  zipFlatIndexGo xs 0 (H := by simp)


-- zipFlatIndex preserves length of the list
theorem List.length_zip_flat_index (xs: List α):  xs.length = (xs.zipFlatIndex).length:= by {
  apply zip_flat_index_go_length;
}

-- The correctness of `List.zipFlatIndex`: value that it zips is the index of the element.
theorem List.zip_flat_index_get (xs: List α) (getIx: Nat) (GETIX: getIx < xs.length):
  (xs.zipFlatIndex[getIx]'(zip_flat_index_length xs ▸ GETIX)).snd.ix = getIx := by {
  sorry
}

-- Map over a tensor with a flattened index
def Tensor.mapWithFlatIndex {σ τ} (v: Tensor σ) (f: TensorFlatIndex (shapeProd v.shape) → σ.eval → τ.eval): Tensor τ :=
  Tensor.mk (shape := v.shape)
    (data := v.data.zipFlatIndex.map (fun (val, ix) => f (v.h_data_size ▸ ix) val)) (h_data_size := by {
   rewrite [List.length_map];
   rewrite [← List.length_zip_flat_index];
   rewrite [v.h_data_size];
   apply Eq.refl;
  })





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

def Vector.size (fixed scalable: List Nat) (scale: List Nat)
    (H: scale.length = scalable.length) :=
  shapeProd fixed * shapeProd (List.map₂ (· * ·) scalable scale)

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
