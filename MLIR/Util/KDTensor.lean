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
  shape0: Nat
  data: List Int --  -> Int
  h_data_size: data.length = shape0

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
def Tensor1D.empty: Tensor1D := { shape0 := 0, data := [], h_data_size :=  rfl }

def Tensor1D.fill (t: Tensor1D) (cst: Int): Tensor1D :=  {
  shape0 := t.shape0
  data := List.replicate t.shape0 cst
  h_data_size := by { simp[List.length_replicate] }
}
theorem List.length_take {α} (as: List α) (len: Nat): (as.take len).length = min as.length len := by sorry

-- Extract upto len `len` from the tensor.
def Tensor1D.extract (t: Tensor1D) (len: Nat): Tensor1D :=
 { 
    shape0 := min t.shape0 len,
    data := t.data.take len,
    h_data_size := by { rewrite [<- t.h_data_size]; apply List.length_take;  }
 }

theorem List.length_drop {α} (as: List α) (k: Nat): (as.drop k).length = as.length - k:= by sorry

-- Offset the indexes into the tensor by `+offset`.
def Tensor1D.offset (t: Tensor1D) (offset: Nat): Tensor1D := {
  shape0 := t.shape0 - offset
  data := t.data.drop offset
  h_data_size := by { rewrite[<- t.h_data_size]; apply List.length_drop; }
}

-- Stride indexes into the tensor by `*stride*.
/-
def Tensor1D.strided (t: Tensor1D) (stride: Nat): Tensor1D := {
  shape0 := t.shape0
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


/-
2D Tensors
-/
structure Tensor2D where
  shape0: Nat
  shape1: Nat
  data: List Int
  h_data_size: data.length = shape0 * shape1

def Tensor2D.isEq (v1 v2: Tensor2D): Decidable (v1 = v2) := by {
  cases v1;
  cases v2;
  simp;
  exact inferInstance;
}

def Tensor2D.empty: Tensor2D :=
  { shape0 := 0, shape1 := 0, data := [], h_data_size := rfl }

instance : Inhabited Tensor2D where
  default := Tensor2D.empty

instance : ToString Tensor2D where
  toString t := "Tensor2D"

/-
4D Tensors
-/
structure Tensor4D where
  shape0: Nat
  shape1: Nat
  shape2: Nat
  shape3: Nat
  data: List Int -- monomorphic tensors
  h_data_size: data.length = (shape0 * shape1 * shape2 * shape3)


def Tensor4D.isEq (v1 v2: Tensor4D): Decidable (v1 = v2) := by {
  cases v1;
  cases v2;
  simp;
  exact inferInstance;
}

def Tensor4D.empty: Tensor4D :=
  { shape0 := 0, shape1 := 0, shape2 := 0, shape3 := 0, data := [], h_data_size := rfl }

instance : Inhabited Tensor4D where
  default := Tensor4D.empty

instance : ToString Tensor4D where
  toString t := "Tensor4D"
