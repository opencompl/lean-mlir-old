/- A toy dialect with basic tensor computations. -/

import MLIRSemantics

def Vector (α: Type) (n: Nat) :=
  Fin n → α

def prod :=
  List.foldl Nat.mul 1

-- Separate the vector and the equality proof so we can define the vector even
-- when dependent pattern matching fails to unify, and prove later
structure Tensor (α: Type) (d: List Nat) where
  size: Nat
  data: Vector α size
  H: size = prod d

/-===========-/
/- Reshaping -/
/-===========-/

def reshape {α d} (d': List Nat) (H: prod d' = prod d):
    Tensor α d → Tensor α d' :=
  λ ⟨n, data, size⟩ => ⟨n, data, by rw [H]; assumption⟩

theorem reshape_reshape {α}:
    ∀ d d' d'' (H': prod d' = prod d) (H'': prod d'' = prod d') (t: Tensor α d),
    reshape d'' H'' (reshape d' H' t) = reshape d'' (Eq.trans H'' H') t := by
  intros d d' d'' H' H'' t;
  simp [reshape]

theorem reshape_same {α}: ∀ d H (t: Tensor α d), reshape d H t = t := by
  intros d H t;
  simp [reshape]

/-===============-/
/- Transposition -/
/-===============-/

def transpose_remap (size n m: Nat) (H: size=n*m): Fin size → Fin size :=
  λ i =>
    let r := i.val / n;
    let j := i.val % n;
    ⟨m*j+r, by sorry /- m*(≤ n-1)+(< m) -/⟩

theorem transpose_remap_involutive (size n m H):
      transpose_remap size m n (by rw [H, Nat.mul_comm])
    ∘ transpose_remap size n m H
    = id := by
  simp [transpose_remap]
  funext i; apply Fin.eq_of_val_eq; simp
  sorry

def transpose {α n m} (t: Tensor α [n,m]): Tensor α [m,n] :=
  ⟨t.size,
   t.data ∘ transpose_remap t.size n m (by simp [t.H, prod, List.foldl]),
   by simp [t.H, prod, List.foldl, Nat.mul_comm]⟩

theorem Function.comp_assoc {α β γ δ} (f: α → β) (g: β → γ) (h: γ → δ):
    (h ∘ g) ∘ f = h ∘ (g ∘ f) :=
  by funext x; simp

theorem transpose_involutive {α n m}:
    ∀ (t: Tensor α [n,m]), transpose (transpose t) = t := by
  intro ⟨size, data, H⟩;
  simp [transpose, Function.comp_assoc, transpose_remap_involutive]
  funext i; simp
