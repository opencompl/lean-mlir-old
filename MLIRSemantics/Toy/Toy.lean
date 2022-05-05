/- A toy dialect with basic tensor computations. -/

import MLIRSemantics.Types
import MLIRSemantics.Util.List

/-
### Tensor reshaping operation

This operation can reshape (retype) tensors with fully-known dimensions,
provided that the number of elements doesn't change.
-/

def reshape {α} {D: DimList} (D': DimList)
    (H: D.known) (H': D'.known) (Hprod: D'.prod = D.prod):
    RankedTensor α D → RankedTensor α D' :=
  fun t =>
    { shape       := D'.project,
      data        := t.data
      h_refines   := dim_known_project_refines H',
      h_data_size := by rw [t.h_data_size, dim_known_prod D' H', Hprod]
                        rw [dim_known_prod_refines H]
                        apply t.h_refines }

theorem reshape_reshape {α} {D: DimList} (D₁ D₂: DimList)
    (H: D.known) (H₁: D₁.known) (H₂: D₂.known)
    (Hprod₁: D₁.prod = D.prod) (Hprod₂: D₂.prod = D₁.prod)
    (t: RankedTensor α D):
      reshape D₂ H₁ H₂ Hprod₂ (reshape D₁ H H₁ Hprod₁ t) =
      reshape D₂ H H₂ (Eq.trans Hprod₂ Hprod₁) t :=
  rfl

theorem reshape_self {α} D H₁ H₂ Hprod (t: RankedTensor α D):
    reshape D H₁ H₂ Hprod t = t := by
  simp [reshape, dim_known_project_eq H₁ t.h_refines]


/-
### Tensor transposition operation

This operation shuffles the elements of the underlying data without changing
its size. To keep this clean it's beneficial to separate the dimension logic
from the index manipulation of the transposition itself.
-/

def transpose_remap (n m: Nat): Nat → Nat :=
  fun i => m * (i % n) + (i / n)

theorem transpose_remap_bound (n m):
    ∀ i, i < n * m → transpose_remap n m i < n * m := by
  intro i h
  simp [transpose_remap]
  sorry /- m*(≤ n-1)+(< m) -/

theorem transpose_remap_involutive (n m):
    ∀i, transpose_remap m n (transpose_remap n m i) = i := by
  simp [transpose_remap, Function.comp]; intro i
  sorry /- = (i/n)*n + i%n -/

@[inline]
def Matrix α n m :=
  RankedTensor α [MLIR.AST.Dimension.Known n, MLIR.AST.Dimension.Known m]

def transpose {α n m} (t: Matrix α n m): Matrix α m n :=
  { shape := [m, n],
    data := List.remap t.data (transpose_remap n m)
        (by intro i h
            rw [t.h_data_size, dim_known_prod_refines _ t.h_refines] at * <;>
            simp at *
            apply transpose_remap_bound; assumption),
    h_refines := by simp,
    h_data_size := by
      simp [shape_prod, List.foldr];
      rw [t.h_data_size, dim_known_prod_refines _ t.h_refines] <;>
      simp [Nat.mul_comm] }

theorem Function.comp_assoc {α β γ δ} (f: α → β) (g: β → γ) (h: γ → δ):
    (h ∘ g) ∘ f = h ∘ (g ∘ f) :=
  by funext x; simp

theorem transpose_involutive {α n m}:
    ∀ (t: Matrix α n m), transpose (transpose t) = t := by
  intro t;
  simp [transpose]
  apply RankedTensor.eq_of_fields_eq <;> simp
  . rw [←dim_known_project_eq _ t.h_refines] <;> simp
  . simp [List.remap_remap]
    apply List.extF <;> simp
    simp [transpose_remap_involutive]
