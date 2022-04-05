import MLIRSemantics.Util.Arith
import Init.Data.List

namespace List

-- General theorems

theorem all_cons {α} (P: α → Bool) head tail:
    all (head::tail) P ↔ P head ∧ all tail P := by
  simp [all, foldr]

theorem all_nil {α} (P: α → Bool):
    all [] P = true := by
  simp [all, foldr]

@[simp]
theorem length_map (as : List α) (f : α → β) :
    (as.map f).length = as.length := by
  induction as with
  | nil => simp [map]
  | cons a as ih => simp [map, ih]

-- getF

def getF {α} (l: List α) (n: Nat) (h: n < l.length): α :=
  match l, n, h with
  | a::as, 0, h => a
  | a::as, n+1, h => getF as n (Nat.lt_of_succ_lt_succ h)

def extF {α} (l₁ l₂: List α) (h_len: l₁.length = l₂.length):
    (∀ (n: Nat) (h: n < l₁.length),
      l₁.getF n h = l₂.getF n (by simp [←h_len, h])) →
    l₁ = l₂ := by
  revert l₂
  induction l₁ with
  | nil =>
      intros l₂ h_len; simp at h_len;
      cases l₂; simp; simp at h_len
  | cons x₁ t₁ ih =>
      intros l₂ h_len; simp at h_len
      cases l₂; simp at h_len
      case cons x₂ t₂ h' =>
        intros h_ext
        have h_x: x₁ = x₂ := by
          specialize (h_ext 0 (by simp; simp_arith));
          simp [getF] at h_ext; apply h_ext
        rw [h_x, ih t₂]
        . simp at h_len; apply h_len
        . intros n h; specialize (h_ext (n+1) (
            by simp; apply Nat.succ_lt_succ; apply h))
          simp [getF] at h_ext; apply h_ext

@[simp]
def getF_map {α β} (l: List α) (f: α → β) n h:
    getF (l.map f) n h = f (getF l n (by simp at h; apply h)) := by
  revert n
  induction l with
  | nil =>
      intros n h
      cases (Nat.not_lt_zero n h)
  | cons a as ih =>
      intros n h
      exact match n with
      | 0 => by simp [getF]
      | m+1 => by simp [getF, ih]

/- theorem getF_eq_get {α} (l: List α) n:
    get? l n.val = some (getF l n) := by -/

-- uniform

def uniform {α} (v: α): Nat → List α
  | 0 => []
  | n+1 => v :: uniform v n

theorem uniform_length {α} (v: α) n: (uniform v n).length = n := by
  induction n <;> simp [uniform]; assumption

-- rangeF

def rangeF (n: Nat): List (Fin n) :=
  match n with
  | 0 => []
  | n+1 => ⟨0,Nat.zero_lt_succ n⟩ ::
      (rangeF n).map (fun i => ⟨i.val+1, Nat.succ_lt_succ i.isLt⟩)

@[simp]
theorem length_rangeF (n: Nat):
    length (rangeF n) = n := by
  induction n with
  | zero => simp
  | succ m ih => simp [rangeF, ih]

@[simp]
theorem getF_rangeF (n: Nat): ∀ i h,
    getF (rangeF n) i h = ⟨i, by simp at h; apply h⟩ := by
  induction n with
  | zero =>
      intro i h;
      cases (Nat.not_lt_zero _ (by simp at h; apply h))
  | succ m ih =>
      intro i
      exact match i with
      | 0 => by simp [getF]
      | j+1 => by simp [rangeF, getF, ih]

--

def remap {α} (l: List α) (f: Nat→Nat) (h: ∀n, n < l.length → f n < l.length) :=
  (rangeF l.length).map (fun i => l.getF (f i.val) (h i.val i.isLt))

@[simp]
theorem length_remap {α} (l: List α) f h:
    (l.remap f h).length = l.length := by
  simp [remap]

@[simp]
theorem getF_remap {α} (l: List α) f h_f n h_n:
    getF (l.remap f h_f) n h_n
    = getF l (f n) (by apply h_f; simp at h_n; apply h_n) := by
  simp [remap]

theorem remap_remap {α} (l: List α) f h_f g h_g:
    remap (remap l f h_f) g h_g
    = remap l (f ∘ g) (fun n h =>
        by simp [Function.comp] at *; apply h_f; apply h_g; apply h) := by
  apply extF <;> simp

end List
