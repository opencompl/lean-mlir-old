import MLIR.Util.Arith
import Init.Data.List

namespace List

-- General theorems

theorem all_cons {α} (P: α → Bool) head tail:
    all (head::tail) P ↔ P head ∧ all tail P := by
  simp [all, foldr]

theorem all_one {α} (P: α → Bool) a:
    all [a] P ↔ P a := by
  simp [all, foldr]

@[simp]
theorem all_nil {α} (P: α → Bool):
    all [] P = true := by
  simp [all, foldr]

@[simp]
theorem length_take {α} (as: List α) (len: Nat):
  (as.take len).length = min as.length len := by sorry

theorem length_take_le (as: List α) (len: Nat):
    (as.take len).length ≤ as.length := by
  simp [length_take]
  apply min_le_left

@[simp]
theorem length_drop {α} (as: List α) (k: Nat):
  (as.drop k).length = as.length - k:= by sorry

@[simp] -- repeat for @[simp]
theorem length_replicate_2 {α} (N: Nat) (a: α):
    length (List.replicate N a) = N :=
  length_replicate _ _

-- getF and reasoning on lists by extensionality

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
      case cons x₂ t₂ =>
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

@[simp]
def getF_replicate {α} (a: α) (N: Nat) (h: n < N):
    getF (List.replicate N a) n (by simp [List.length_replicate, h]) = a := by
  revert n
  induction N with
  | zero => intros n h; cases h
  | succ m ih =>
      intros n h
      simp [replicate]
      cases n <;> simp [getF]
      rw [ih]
      apply Nat.lt_of_succ_lt_succ; assumption

@[simp]
def getF_take {α: Type} {l: List α} {N n: Nat} (h: n < length (take N l)):
    getF (List.take N l) n h = getF l n (by
      simp [length_take] at h
      apply Nat.lt_of_lt_of_le h
      apply Nat.min_le_left) := by
  revert n l
  induction N with
  | zero => intros l n h; cases h
  | succ m ih =>
      intros l n h
      cases l with
      | nil => simp [take]
      | cons head tail =>
          simp [take]
          cases n <;> simp [getF, ih]

theorem List.getF_implies_mem: ∀ {α: Type} (xs: List α) (i: Nat) (INBOUND: i < xs.length),
 List.Mem (List.getF xs i INBOUND) xs := by {
  intros α xs;
  induction xs;
  case nil => {
    intros i INBOUND; simp at INBOUND;
    simp [Nat.not_lt_zero] at INBOUND;
  }
  case cons x' xs IH => {
    intros i INBOUND;
    cases i;
    case zero => {
       simp [List.getF];
       constructor;
    }
    case succ i' => {
     simp [List.getF];
     constructor;
     apply IH;
   }
  }
}


/- theorem getF_eq_get {α} (l: List α) n:
    get? l n.val = some (getF l n) := by -/

-- uniform

def uniform {α} (v: α): Nat → List α
  | 0 => []
  | n+1 => v :: uniform v n

theorem length_uniform {α} (v: α) n: (uniform v n).length = n := by
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
