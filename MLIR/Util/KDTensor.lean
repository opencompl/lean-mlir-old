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
  data: List Int
  h_data_size: data.length = shape0

def Tensor1D.isEq (v1 v2: Tensor1D): Decidable (v1 = v2) := by {
  cases v1;
  cases v2;
  simp;
  exact inferInstance;
}

def Tensor1D.empty: Tensor1D :=
  { shape0 := 0, data := [], h_data_size := rfl }

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
