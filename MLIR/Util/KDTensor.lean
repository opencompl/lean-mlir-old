import MLIR.Util.Mathlib4.NatBasic
import MLIR.Util.Mathlib4.Dvd
import MLIR.Util.Mathlib4.NatLemmas
import MLIR.Util.List
import MLIR.Util.FinInt

/-
This file defines the theory of K-dimensional arrays (for fixed K=4).
This aspires to be generalized to arbitrary dimensions, but for now,
we develop the theory for fixed dimension.

TODO: please unify:
- MLIR/Model/BuiltinModel.lean
- MLIR/Util/KDTensor.lean
- MLIR/Semantics/TensorElem.lean
-/

structure Tensor1D where
  size0: Nat
  data: List (FinInt 32) --  -> Int
  h_data_size: data.length = size0

def Tensor1D.isEq (v1 v2: Tensor1D): Decidable (v1 = v2) := by {
  cases v1;
  cases v2;
  simp;
  exact inferInstance;
}


/-
### Primops that manipulate tensors.

These primitive operations are *constructive*, in that they build
simple tensors from other tensors by either manipulating the shape XOR the data,
never both. Decomposing other tensor transforms into these primitives
allows us to decompose the shape reasoning from the data reasoning.

All other operations must be written in terms of these primitives.
-/
def Tensor1D.empty: Tensor1D := { size0 := 0, data := [], h_data_size :=  rfl }

def Tensor1D.fill (t: Tensor1D) (cst: FinInt 32): Tensor1D :=  {
  size0 := t.size0
  data := List.replicate t.size0 cst
  h_data_size := by { simp[List.length_replicate] }
}

-- Extract upto len `len` from the tensor.
def Tensor1D.extract (t: Tensor1D) (len: Nat): Tensor1D :=
 { 
    size0 := min t.size0 len,
    data := t.data.take len,
    h_data_size := by { rewrite [<- t.h_data_size]; apply List.length_take;  }
 }

-- Offset the indexes into the tensor by `+offset`.
def Tensor1D.offset (t: Tensor1D) (offset: Nat): Tensor1D := {
  size0 := t.size0 - offset
  data := t.data.drop offset
  h_data_size := by { rewrite[<- t.h_data_size]; apply List.length_drop; }
}

-- Stride indexes into the tensor by `*stride*.
/-
def Tensor1D.strided (t: Tensor1D) (stride: Nat): Tensor1D := {
  size0 := t.size0
  data := fun n => t.data (n * stride)
}
-/


/-
TODO: Build a theory that shows how to modify the *index* to be equivalent to the operation
on the *tensor*.
-/


instance : Inhabited Tensor1D where
  default := Tensor1D.empty

instance : ToString Tensor1D where
  toString t := "Tensor1D"


structure Point2D where 
  x: Nat
  y: Nat 


structure TensorIndex2D (size0: Nat) (size1: Nat): Type where 
  ix0: Nat
  ix1: Nat 
  IX0: ix0 < size0
  IX1: ix1 < size1

/-
A subview into a 2D tensor.
-/
structure TensorSubview2D (maxsize0: Nat) (maxsize1: Nat): Type where 
  -- ix0: Nat 
  -- ix1: Nat 
  size0: Nat 
  size1: Nat 
  IX0: size0 <= maxsize0 
  IX1: size1 <= maxsize1


  


-- enlarge the tensor index to index a larger space.
def TensorIndex2D.enlarge {size0 size0' size1 size1': Nat}
  (INC0: size0 <= size0') (INC1: size1 <= size1')
  (ix: TensorIndex2D size0 size1): TensorIndex2D size0' size1' := {
    ix0 := ix.ix0
    ix1 := ix.ix1
    IX0 := by {  
        have H: ix.ix0 < size0 := ix.IX0;
        simp_arith;
        apply Nat.lt_of_lt_of_le H INC0;
        }
    IX1 := by {
        have H: ix.ix1 < size1 := ix.IX1;
        simp_arith;
        apply Nat.lt_of_lt_of_le H INC1;
    }
  }
-- extract subview: (n, m) of (N, M)
-- subview: TensorSubview2D n m
-- tensor: Tensor N M
-- hypothesis: (n <= N) (m <= M)
-- produce: Tensor n m
--   ix => TensorIndex2D n m
--     ix.size0 < n
--     ix.size1 < m


def TensorIndex2D.transpose
  (ix: TensorIndex2D size0 size1): TensorIndex2D size1 size0 := {
    ix0 := ix.ix1
    ix1 := ix.ix0
    IX0 := ix.IX1
    IX1 := ix.IX0
  }

lemma Nat.lt_mul_cancel_left (a b x: Nat) (H: a < b): a * x < b * x := by sorry;

def TensorIndex2D.stride (ix: TensorIndex2D size0 size1) (stride0: Nat) (stride1: Nat):
  TensorIndex2D (size0*stride0) (size1*stride1) := {
  ix0 := ix.ix0 * stride0
  ix1 := ix.ix1 * stride1
  IX0 := by { 
      have H: ix.ix0 < size0 := ix.IX0;
      simp [H, Nat.lt_mul_cancel_left]; 
     }
  IX1 := by { 
      have H: ix.ix1 < size1 := ix.IX1;
      simp [H, Nat.lt_mul_cancel_left]; 
  }
}
/-
2D Tensors
-/
structure Tensor2D where
  size0: Nat
  size1: Nat
  /- Switch to using TensorIndex? -/
  data: TensorIndex2D size0 size1 -> Int 

def Tensor2D.isEq (v1 v2: Tensor2D): Decidable (v1 = v2) :=
  .isTrue sorry -- FIXME: check if we ever use decidable equality 

def Tensor2D.empty: Tensor2D :=
  { size0 := 0, size1 := 0, data := fun ix => by {
      have CONTRA: ix.ix0 < 0 := ix.IX0;
      simp[Nat.not_lt_zero] at CONTRA;
    }
  }


/-
from a subview of size nxm, extract out a tensor of size nxm,
given a larger tensor of size (t.size0 x t.size1)
-/
def TensorSubview2D.extract (view: TensorSubview2D n m)
  (t: Tensor2D)
  (HN: n <= t.size0) (HM: m <= t.size1): Tensor2D  := 
  Tensor2D.mk view.size0 view.size1
    (fun ix => t.data (ix.enlarge 
        (by {
          have HMID : view.size0 <= n := view.IX0;
          apply Nat.le_trans;
          apply HMID;
          apply HN;
        }) (by {
          have HMID : view.size1 <= m := view.IX1;
          apply Nat.le_trans;
          apply HMID;
          apply HM;
        })))

def Tensor2D.extractSubview (t: Tensor2D) (subview: TensorSubview2D t.size0 t.size1):
  Tensor2D  := Tensor2D.mk subview.size0 subview.size1 
    (fun ix => (subview.extract t (by simp) (by simp)).data ix )

instance : Inhabited Tensor2D where
  default := Tensor2D.empty

instance : ToString Tensor2D where
  toString t := "Tensor2D"


/-
Create a tensor2d filled with the same value.
-/
def Tensor2D.fill (t: Tensor2D) (val: Int): Tensor2D :=
  Tensor2D.mk t.size0 t.size1 (fun _ix => val)

def Tensor2D.extractslice
  (t: Tensor2D)
  (size0 size1: Nat)
  (SIZE0: size0 <= t.size0) (SIZE1: size1 <= t.size1): Tensor2D :=
   Tensor2D.mk size0 size1
    (fun ix => t.data (ix.enlarge SIZE0 SIZE1))


def Tensor2D.extractslice' (large: Tensor2D)
  (subview: TensorSubview2D large.size0 large.size1): Tensor2D :=
  Tensor2D.mk subview.size0 subview.size1 
    (fun ix => large.data (ix.enlarge subview.IX0 subview.IX1))

-- Transpose of a tensor by swapping indexes
def Tensor2D.transpose (t: Tensor2D): Tensor2D :=
  Tensor2D.mk t.size1 t.size0 (fun ix => t.data ix.transpose)

-- Stride index into a tensor, scaling the indexing by `stride0, stride1`. 
def Tensor2D.stride (t: Tensor2D) (stride0 stride1: Nat)
  (STRIDE0: stride0 > 0) (STRIDE1: stride1 > 0): Tensor2D :=
  Tensor2D.mk (t.size0 / stride0) (t.size1 / stride1) 
    (fun ix => t.data <| (ix.stride stride0 stride1).enlarge
    (by { rewrite[<- Nat.le_div_iff_mul_le]; simp_arith; apply STRIDE0; })
    (by { rewrite[<- Nat.le_div_iff_mul_le]; simp_arith; apply STRIDE1; }))


def Tensor2D.toSubview (t: Tensor2D): TensorSubview2D t.size0 t.size1 :=  {
    size0 := t.size0,
    size1 := t.size1,
    IX0 := by simp,
    IX1 := by simp,
  }


def TensorIndex2D.isInSubview (t: Tensor2D) (subview: TensorSubview2D t.size0 t.size1)
  (ix: TensorIndex2D t.size0 t.size1): 
  Option (TensorIndex2D subview.size0 subview.size1) :=
  dite (ix.ix0 < subview.size0)
  (fun LT0 => 
    dite (ix.ix1 < subview.size1)
    (fun LT1 => 
      .some (TensorIndex2D.mk ix.ix0 ix.ix1 LT0 LT1)
    )
    (fun GEQ1 => .none))
  (fun GEQ0 => .none)

/-
Represents that `small` is located inside a slice of `large`.
-/
structure TensorSlice2D (small large: Tensor2D) where 
  SIZE0: small.size0 <= large.size0
  SIZE1: small.size1 <= large.size1

def TensorSlice2D.toSubview (slice: TensorSlice2D small large): 
  TensorSubview2D large.size0 large.size1 := {
    size0 := small.size0,
    size1 := small.size1,
    IX0 := slice.SIZE0,
    IX1 := slice.SIZE1 }

/-
4D Tensors
-/
structure Tensor4D where
  size0: Nat
  size1: Nat
  shape2: Nat
  shape3: Nat
  data: List Int -- monomorphic tensors
  h_data_size: data.length = (size0 * size1 * shape2 * shape3)


def Tensor4D.isEq (v1 v2: Tensor4D): Decidable (v1 = v2) := by {
  cases v1;
  cases v2;
  simp;
  exact inferInstance;
}

def Tensor4D.empty: Tensor4D :=
  { size0 := 0, size1 := 0, shape2 := 0, shape3 := 0, data := [], h_data_size := rfl }

instance : Inhabited Tensor4D where
  default := Tensor4D.empty

instance : ToString Tensor4D where
  toString t := "Tensor4D"


/-
### shapeProd
-/

def shapeProd: List Nat → Nat :=
  List.foldr (·*·) 1

theorem shape_prod_nil: shapeProd (0::l) = 0 := by
  induction l <;> simp [shapeProd, List.foldr]

@[simp]
theorem shapeProd.cons_unfold: ∀ (x: Nat) (xs: List Nat),
  shapeProd (x :: xs) = x * shapeProd xs := by {
   intros x xs;
   simp [shapeProd, List.foldr];
}

/-
### Flat tensor index
-/
/-
A 1D index into a tensor. Witnesses that the flat index is in bounds of the shape of the tensor.
-/
structure TensorFlatIndex (bound: Nat) where
  ix: Nat
  h_ix_inbound: (ix < bound)

def TensorFlatIndex.eq_proof_irrelevant  (f1: TensorFlatIndex b) (f2: TensorFlatIndex b) (IXEQ: f1.ix = f2.ix): f1 = f2 := by {
  induction f1;
  case mk ix1 H1 => {
  induction f2;
  case mk ix2 H2 => {
   simp [IXEQ];
  }
  }
}


def TensorFlatIndex.cast_left: ∀ (bound bound': ℕ) (EQ: bound = bound') (ix: ℕ) (prf: ix < bound) (prf': ix < bound'),
  EQ ▸ { ix := ix, h_ix_inbound := prf : TensorFlatIndex bound } = {ix := ix, h_ix_inbound := prf' }
   := by {
  intros bound bound';
  intros EQ ix prf prf';
  cases EQ;
  simp;
}

def TensorFlatIndex.cast_right: ∀ (bound bound': ℕ) (EQ: bound = bound') (ix: ℕ) (prf: ix < bound) (prf': ix < bound'),
  { ix := ix, h_ix_inbound := prf : TensorFlatIndex bound } = EQ ▸ {ix := ix, h_ix_inbound := prf' }
   := by {
  intros bound bound';
  intros EQ ix prf prf';
  cases EQ;
  simp;
}

theorem TensorFlatIndex.bound_non_zero (flat: TensorFlatIndex bound): bound ≠ 0 := by {
  intros BOUND;
  have H_INBOUND := flat.h_ix_inbound;
  simp [BOUND] at H_INBOUND;
  simp [Nat.not_lt_zero] at H_INBOUND;
}

theorem TensorFlatIndex.bound_zero_absurd (flat: TensorFlatIndex 0): False := by {
  have H_INBOUND := flat.h_ix_inbound;
  simp [Nat.not_lt_zero] at H_INBOUND;
}

@[simp]
theorem Nat.succ_gt_zero (n: Nat): Nat.succ n > 0 := by {
  simp [GT.gt];
  simp [Nat.zero_lt_succ];
}

@[simp]
theorem Nat.nonzero_iff_gt_zero: ∀ (n: Nat), n ≠ 0 <-> n > 0 := by {
  intros n;
  constructor;
  case mp => {
  intros NEQ_0;
  cases n;
  case zero => {
    contradiction;
  }
  case succ n' => { simp [Nat.succ_gt_zero]; }
  }
  case mpr => {
   intros GT_ZERO;
   cases n;
   case zero => {
     simp at GT_ZERO;
   }
   case succ n' => { simp; }
  }
}

-- Bound is always greater than zero.
theorem TensorFlatIndex.bound_gt_zero(flat: TensorFlatIndex bound): bound > 0 := by {
  have BOUND_NONZERO: bound ≠ 0 := TensorFlatIndex.bound_non_zero flat;
  cases bound;
  case zero => {
    simp [Nat.zero, BOUND_NONZERO];
    contradiction;
  }
  case succ bound' => {
    apply Nat.succ_gt_zero;
  }
}

@[simp]
theorem Nat.mul_nonzero_implies_left_nonzero: ∀ (a b: Nat) (NEQ: a * b ≠ 0), a ≠ 0 := by {
  intros a b NEQ;
  induction a;
  case zero => {
   simp at NEQ;
  }
  case succ a' IH => {
    apply Nat.succ_ne_zero;
  }
}

@[simp]
theorem Nat.mul_nonzero_implies_right_nonzero: ∀ (a b : Nat) (NEQ: a * b ≠ 0), b ≠ 0 := by {
  intros a b NEQ;
  induction a;
  case zero => {
   simp at NEQ;
  }
  case succ a' IH => {
    induction b;
    case zero => {
     simp at NEQ;
    }
    case succ b' IH => {
     apply Nat.succ_ne_zero;
    }
  }
}

-- if product of number is nonzero, then every element is nonzero
theorem shapeProd_nonzero_implies_member_nonzero: ∀ (xs: List Nat)
   (x: Nat) (MEM: List.Mem x xs) (PROD: shapeProd xs > 0) , x > 0 := by {
   intros xs x MEM;
   induction MEM;
   case head a as => {
     simp [shapeProd, List.foldr];
     intros H;
     rewrite [<- Nat.nonzero_iff_gt_zero];
     apply Nat.mul_nonzero_implies_left_nonzero;
     rewrite [<- Nat.nonzero_iff_gt_zero] at H;
     apply H;
   }
   case tail b bs MEM IH => {
     intros H;
     apply IH;
     simp at H;
     rewrite [<- Nat.nonzero_iff_gt_zero] at *;
     apply (Nat.mul_nonzero_implies_right_nonzero);
     apply H;
   }
}


-- A TensorFlatIndex of a shapeProd will be nonzero.
theorem TensorFlatIndex.shapeProd_member_nonzero
  (shape: List Nat)
  (flat: TensorFlatIndex (shapeProd shape))
  (n: Nat) (MEMBER: List.Mem n shape): n > 0 := by {
  have PROD_NONZERO: shapeProd shape > 0 := flat.bound_gt_zero;
  apply shapeProd_nonzero_implies_member_nonzero;
  exact MEMBER;
  exact PROD_NONZERO;
}


@[simp]
theorem Nat.mod_zero_implies_div_mul_equal (n: Nat) (modulus: Nat)
  (MODZERO: n % modulus = 0): (n / modulus) * modulus = n := by {
  have MULTIPLE: n = 0 + (n / modulus) * modulus := by {
    rewrite [<- MODZERO];
    rewrite [Nat.mul_comm];
    simp [Nat.mod_add_div];
  }
  simp at MULTIPLE;
  rewrite [<- MULTIPLE];
  rfl;
}

@[simp]
theorem Nat.mul_cancel_right (n m: Nat) (MODZERO: n % m = 0): (n / m) * m = n := by {
    rewrite [Nat.mod_zero_implies_div_mul_equal n m MODZERO];
    rfl;
}

@[simp]
theorem Nat.div_lt_if_mod (ix bound modulus: Nat) (IX: ix < bound) (MODULUS: modulus > 0) (DIV: bound % modulus = 0):
  ix / modulus < bound / modulus := by {
  rewrite [Nat.div_lt_iff_lt_mul, Nat.mul_cancel_right];
  apply IX;
  apply DIV;
  apply MODULUS;
}

-- A theory of splitting and merging
-- 'TensorFlatIndex'es. This will be used to provide a theory
-- of delinearizing arbitrary tensor indexes
-- in terms of TensorFlatIndexes.
-- Split a TensorFlatIndex into two
def TensorFlatIndex.split
  (n modulus: Nat) (MODULUS: modulus > 0) (DIV: n % modulus = 0)
  (flat: TensorFlatIndex n): (TensorFlatIndex modulus) × (TensorFlatIndex (n/modulus)) :=
  (TensorFlatIndex.mk (flat.ix %  modulus) (Nat.mod_lt flat.ix MODULUS),
   TensorFlatIndex.mk (flat.ix / modulus) (Nat.div_lt_if_mod flat.ix n modulus flat.h_ix_inbound MODULUS DIV))

theorem Nat.le_pred_if_lt (x n : Nat) (X_LT_N: x < n): x <= pred n := by {
     cases n;
     case zero => { simp [Nat.not_lt_zero] at X_LT_N; }
     case succ n' => {
      rewrite [Nat.pred_succ];
      apply Nat.le_of_lt_succ;
      exact X_LT_N;
    }
}

theorem Nat.le_one_minus_if_lt (x n : Nat) (X_LT_N: x < n): x <= pred n := by {
    rewrite [<- Nat.sub_one];
    apply Nat.le_pred_if_lt;
    simp; exact X_LT_N;
}

theorem Nat.le_mul_pred (x y n: Nat) (LE: x <= Nat.pred n): x * y <= n * y - y := by {
   cases H:n;
   case zero => {
   rewrite [H] at LE;
   simp at LE;
   rewrite [LE];
   simp;
   }
   case succ n' => {
   simp at LE;
   rewrite [H] at LE;
   simp at LE;
   sorry; -- algebra to be done.
   }
}

-- x < n <=> x <= n - 1
-- #check Nat.lt_of_succ_le
-- Merge a TensorFlatIndex into a large TensorFlatIndex
def TensorFlatIndex.merge
  (flat0: TensorFlatIndex N0)
  (flat1: TensorFlatIndex N1): TensorFlatIndex (N0 * N1) :=
  TensorFlatIndex.mk (flat1.ix * N1 + flat0.ix) (by {
     have IX0: flat0.ix <= Nat.pred N0 := Nat.le_pred_if_lt _ _ flat0.h_ix_inbound;
     have IX1: flat1.ix <= Nat.pred N1 := Nat.le_pred_if_lt _ _ flat1.h_ix_inbound;
     have IX0_N: flat0.ix * N1 <= N0 * N1 - N1 := by {
      apply Nat.le_mul_pred <;> simp;
      exact IX0;
     }
     -- algebra
     sorry
  })






inductive TensorIndex': List Nat -> Type :=
|  Empty: TensorIndex' []
|  Dim (bound0: Nat)
      (ix: TensorFlatIndex bound0)
      (rest: TensorIndex' shape): TensorIndex' (bound0 :: shape)


/-
Projecting out outermost dimension
-/
def TensorIndex'.projectOut
  {outermost: Nat}
  {shape: List Nat}
  (index: TensorIndex' (outermost :: shape)): TensorIndex' shape :=
  match index with
  | .Dim _ _ rest => rest

inductive List.NonEmpty: List α-> Prop where
| Singleton (a: α): List.NonEmpty [a]
| Cons (a: α) (AS: List.NonEmpty as): List.NonEmpty (a::as)


theorem List.NonEmpty.empty_absurd (α: Type) (CONTRA: List.NonEmpty (@List.nil α)): False := by {
  cases CONTRA;
}

@[simp]
theorem TensorIndex'.empty_dims_is_empty (index: TensorIndex' []): index = .Empty := by {
  cases index; simp;
}

@[reducible, simp]
def TensorIndex'.getLinearizedIndexNumber
   {dims: List Nat} (index: TensorIndex' dims) : TensorFlatIndex (shapeProd dims) :=
    match index with
    | .Empty =>  TensorFlatIndex.mk 0 (by {simp[shapeProd];})
    | .Dim bound0 ix rest => ix.merge rest.getLinearizedIndexNumber


theorem Nat.lt_iff_gt: ∀ (a: Nat) (b: Nat), a < b <-> b > a := by {
  intros a b; constructor;
  case mp => {
     intros A_LT_B;
     simp [GT.gt]; exact A_LT_B;
  }
  case mpr => {
    intros B_GT_A;
    simp [GT.gt] at B_GT_A;
    exact B_GT_A;
  }
}



/-
### Naivete of definition of delineralizatoin

One might initially choose to believe that for ANY modulus, we can delin
def TensorIndex.delinearizeInnermost {innerDim: Nat} {restDims: List Nat}
  (modulus: Nat)
  (index: TensorIndex (innerDim :: restDims)):
    TensorIndex (modulus :: (innerDim/modulus) :: restDims) :=

This is absurd, because I Can choose modulus to be super large (9999), then
the tensor collapses because (innerDim/modulus) becomes = 0.


As a second try, one can try to add the assumtion that (modulus < innerDim).
This too is insufficient!
For the shape:
  (modulus, innerDim / modulus, ...)
we would naively choose the indexes:
  (innermostix % modulus, innermostix / modulus)

The 0th entry is clearly inbounds:
  (innermostix % modulus) < modulus

the 1st entry is not necessarily inbounds!
    innermostix / modulus < innerDim / modulus ??
   Even given that (innermostix < innerDim) from the original tensor, we cannot
   conclude that division preserves less than!
   eg 2 < 3 =/=> (2/9999) < (3/9999)!


We need some kind of divisibility criterion.
-/

theorem shapeProd_cons_prod (x y: Nat) (zs: List Nat): shapeProd (x :: y :: zs) = shapeProd ((x *y) :: zs) := by {
   simp [Nat.mul_assoc];
}


-- Build a 1D TensorIndex from a FlatIndex
def TensorIndex'.ofFlatIndex1D {innerDim: Nat}
  (flat: TensorFlatIndex innerDim): TensorIndex' [innerDim] := .Dim innerDim flat .Empty

theorem Nat.mul_of_nonzero_is_nonzero: ∀ (a b: Nat) (A: a ≠ 0) (B: b ≠ 0), a * b ≠ 0 := by {
   intros a;
   induction a;
   case zero => {
     intros b A_NEQ_ZERO; simp [A_NEQ_ZERO]; contradiction;
   }
   case succ a' IH => {
     intros b;
     induction b;
     case zero => {
        intros A B;
        simp at B;
     }
     case succ b' IH' => {
      intros A B;
      simp [Nat.mul];
      rewrite [Nat.nonzero_iff_gt_zero] at *;
      simp [Nat.mul_pos];
    }
   }

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
     }
     let ix' := ix + 1
     let H' :ix' + xs'.length = bound := by {
       rewrite [← H];
       simp;
       rewrite [Nat.succ_eq_add_one];
       -- ⊢ ix + 1 + List.length xs' = ix + (List.length xs' + 1)
       have SWIZZLE : (1 + List.length xs' = List.length xs' + 1) := by simp[Nat.add_comm];
       rewrite [Nat.add_assoc];
       rewrite [SWIZZLE];
       simp;
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


-- The value of the (zipFlatIndexGo xs ix bound ...):
--   ie, we have a list of total length `bound`, we have read list upto index `ix`, and the rest of the list is `xs`,
--   must be (ix + deltaIx).
theorem List.zip_flat_index_go_get: ∀ (xs: List α) (ix: Nat) (bound: Nat) (H: ix + xs.length = bound)
  (deltaIx: Nat) (GETIX: deltaIx < xs.length),
  ((zipFlatIndexGo xs ix bound H).getF deltaIx (zip_flat_index_go_length xs ix bound H ▸ GETIX)) =
  (xs.getF deltaIx GETIX, TensorFlatIndex.mk (bound := bound)
                           (ix := ix + deltaIx)
                           (h_ix_inbound := by { rewrite [<- H]; simp [Nat.add_lt_add_left, GETIX]; } )) := by {
  intros xs;
  induction xs;
  case nil => {
      intros ix bound H deltaIx GETIX;
      simp [List.length, Nat.not_lt_zero] at GETIX;
  }
  case cons x xs' IND => {
   intros ix bound H deltaIx GETIX; -- consider pulling deltaIx earlier
   cases deltaIx;
   case zero => {
      simp;
      simp [zipFlatIndexGo, List.getF]
   }
   case succ deltaIx' => {
     simp [zipFlatIndexGo];
     simp [List.getF];
     rewrite [IND];
     simp [Nat.add_assoc, Nat.add_one, Nat.succ_add, Nat.add_succ];
     simp at GETIX;
     apply Nat.lt_of_succ_lt_succ;
     exact GETIX;
   }
  }
}

-- Zip a list with the index of the current value
def List.zipFlatIndex (xs: List α): List (α × TensorFlatIndex xs.length) :=
  zipFlatIndexGo xs 0 (H := by simp)


-- zipFlatIndex preserves length of the list
@[simp]
theorem List.length_zip_flat_index (xs: List α): length (List.zipFlatIndex xs) = length xs := by {
  apply Eq.symm;
  apply zip_flat_index_go_length;
}

-- The correctness of `List.zipFlatIndex`: value that it zips is the index of the element.
theorem List.zip_flat_index_get (xs: List α) (getIx: Nat) (GETIX: getIx < xs.length):
  (List.getF (List.zipFlatIndex xs) getIx (by simp; apply GETIX)) = (List.getF xs getIx GETIX, TensorFlatIndex.mk (bound := xs.length) getIx GETIX) := by {
  simp[zipFlatIndex];
  have RHS :  { ix := getIx, h_ix_inbound := GETIX : TensorFlatIndex (xs.length) } = {ix := 0 + getIx, h_ix_inbound := by { simp; apply GETIX } : TensorFlatIndex (xs.length)} := by {
    simp;
  }
  rewrite [RHS];
  apply List.zip_flat_index_go_get (xs := xs) (ix := 0) (bound := List.length xs) (deltaIx := getIx) (GETIX := GETIX);
}


def Tensor1D.mapWithFlatIndex (v: Tensor1D) (f: TensorFlatIndex v.size0 →  (FinInt 32) →  (FinInt 32)):
  Tensor1D :=
  Tensor1D.mk (size0 := v.size0)
    (data := (List.zipFlatIndex v.data).map (fun (val, ix) => f (v.h_data_size ▸ ix) val)) (h_data_size := by simp; apply v.h_data_size)

def Tensor1D.mapMWithFlatIndex {M: Type -> Type} [Monad M]
  (v: Tensor1D) (f: TensorFlatIndex v.size0 → (FinInt 32) → M (FinInt 32)):
  M Tensor1D := do
  let data <-
      (List.zipFlatIndex v.data).mapM (fun (val, ix) => f (v.h_data_size ▸ ix) val)
  let temp := Tensor1D.mk data.length data rfl
  return temp

theorem List.mapM_loop_map [Monad M] [LawfulMonad M]
    (l: List α) (f: α → β) (fM: α → M β) (results: List β):
    (forall a, fM a = return f a) →
    List.mapM.loop fM l results = return results.reverse ++ l.map f := by
  intros h
  revert results
  induction l with
  | nil => intros results; simp [map]; rfl
  | cons a l ih =>
      intros results
      simp [mapM.loop, map, h, ih, reverse_cons, append_assoc]

theorem List.mapM_map [Monad M] [LawfulMonad M] (l: List α) (f: α → β) (fM: α → M β):
    (forall a, fM a = return f a) →
    l.mapM fM = return l.map f := by
  apply List.mapM_loop_map

theorem Tensor1D.mapM_map [Monad M] [LawfulMonad M] v f (fM: _ → _ → M _):
    (forall flat_index val, fM flat_index val = return f flat_index val) →
    mapMWithFlatIndex v fM = return mapWithFlatIndex v f := by
  intros h
  unfold mapWithFlatIndex
  unfold mapMWithFlatIndex
  rw [List.mapM_map]
  . simp [v.h_data_size]; rfl
  . intros a; cases a; simp [h]
