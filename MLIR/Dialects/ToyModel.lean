/- A toy dialect with basic tensor computations. -/

import MLIR.Dialects.BuiltinModel
import MLIR.Util.List

open DimList
open MLIR.AST (TensorElem)


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
def Matrix n m τ :=
  RankedTensor [MLIR.AST.Dimension.Known n, MLIR.AST.Dimension.Known m] τ

def transpose (t: Matrix n m τ): Matrix m n τ :=
  { shape := [m, n],
    data := List.remap t.data (transpose_remap n m)
        (by intro i h
            rw [t.h_data_size, dim_known_prod_refines _ t.h_refines] at * <;>
            simp at *
            apply transpose_remap_bound; assumption),
    h_refines := by simp,
    h_data_size := by
      simp [TensorElem.shapeProd, List.foldr];
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
