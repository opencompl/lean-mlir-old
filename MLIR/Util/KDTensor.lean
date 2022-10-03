/-
This file defines the theory of K-dimensional arrays (for fixed K=4).
This aspires to be generalized to arbitrary dimensions, but for now,
we develop the theory for fixed dimension.
-/

structure KDTensor where
  shape0: Nat
  shape1: Nat
  shape2: Nat
  shape3: Nat
  data: List Int -- monomorphic tensors
  h_data_size: data.length = (shape0 * shape1 * shape2 * shape3)


def KDTensor.isEq (v1 v2: KDTensor): Decidable (v1 = v2) := by {
  cases v1;
  cases v2;
  simp;
  exact inferInstance;
}

def KDTensor.empty: KDTensor :=
  { shape0 := 0, shape1 := 0, shape2 := 0, shape3 := 0, data := [], h_data_size := rfl }

instance : Inhabited KDTensor where
  default := KDTensor.empty

instance : ToString KDTensor where
  toString t := "KDTensor"
