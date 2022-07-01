/-
### Addition and subtraction of Int
-/

namespace Int

@[simp]
theorem sub_zero (n: Int): n - 0 = n := by
  cases n <;> rfl

@[simp]
theorem add_assoc (n m k: Int): n + (m + k) = n + m + k := by
  sorry

theorem add_comm (n m: Int): n + m = m + n := by
  simp [HAdd.hAdd, Add.add, Int.add]
  cases n <;> cases m <;> simp [Nat.add_comm]

@[simp]
theorem add_zero (n: Int): n + 0 = n := by
  simp [HAdd.hAdd, Add.add, Int.add]
  cases n <;> simp; rfl

theorem add_sub_assoc (n m k: Int): n + m - k = n + (m - k) := by
  sorry

theorem add_sub (n m: Int): n + (m - n) = m := by
  sorry

theorem sub_add_dist (n m p: Int): n - (m + p) = n - m - p := by
  sorry

/-
### Multiplication and power of Int
-/

theorem mul_comm (n m: Int): n * m = m * n := by
  simp [HMul.hMul, Mul.mul, Int.mul]
  cases n <;> cases m <;> simp [Nat.mul_comm]

theorem mul_add (n m k: Int): n * (m + k) = n * m + n * k := by
  sorry

@[simp]
theorem mul_zero (n: Int): n * 0 = 0 := by
  simp [HMul.hMul, Mul.mul, Int.mul]
  cases n <;> rfl

@[simp]
theorem mul_one (n: Int): n * 1 = n := by
  simp [HMul.hMul, Mul.mul, Int.mul]
  cases n <;> simp; rfl

theorem mul_two (n: Int): n * 2 = n + n := by
  have h: (2:Int) = 1 + 1 := rfl
  simp [h, mul_add, mul_one]

@[simp]
theorem pow_zero (n: Int): n^0 = 1 := by rfl

theorem pow_succ (n: Int) (m: Nat): n^(m+1) = n^m * n := by rfl

/-
### Order of Int
-/

theorem ge_zero_eq_nonneg (n: Int): n ≥ 0 ↔ Int.NonNeg n := by
  simp [GE.ge, LE.le, Int.le]

theorem add_ge_zero (n m: Int): n ≥ 0 → m ≥ 0 → n + m ≥ 0 := by
  simp [GE.ge, LE.le, Int.le, HAdd.hAdd, Add.add, Int.add]
  intros hn hm
  cases hn; cases hm; simp; constructor

theorem le_succ (n: Int): n ≤ n+1 := by
  suffices NonNeg 1 by simp [LE.le, Int.le, add_sub_assoc, add_sub]; trivial
  constructor

theorem succ_le_succ (n m: Int): n ≤ m → n + 1 ≤ m + 1 := by
  sorry

theorem le_trans {n k} (m: Int): n ≤ m → m ≤ k → n ≤ k := fun h₁ h₂ => by
  suffices NonNeg (k-n) by trivial
  simp [GE.ge, LE.le, Int.le, ←ge_zero_eq_nonneg] at *
  suffices (k - m) + (m - n) ≥ 0 by sorry
  apply add_ge_zero <;> assumption

theorem lt_trans {n k} (m: Int): n < m → m < k → n < k := fun h₁ h₂ => by
  simp [LT.lt, Int.lt] at *
  have h: (2:Int) = 1 + 1 := rfl
  apply le_trans (n+2)
  . simp [h]; apply le_succ
  . apply le_trans (m+1) _ h₂; simp [h]; apply succ_le_succ; trivial

theorem lt_add_right (n m: Int): m > 0 → n < n + m := by
  sorry

theorem mul_ge_zero {n m: Int}: n ≥ 0 → m ≥ 0 → n * m ≥ 0 := by
  rw [Int.ge_zero_eq_nonneg, Int.ge_zero_eq_nonneg, Int.ge_zero_eq_nonneg]
  intros hn hm;
  cases hn; cases hm; constructor

theorem mul_gt_zero {n m: Int}: n > 0 → m > 0 → n * m > 0 := by
  sorry

theorem pow_ge_zero (n: Int) (m: Nat): n ≥ 0 → n^m ≥ 0 := by
  revert n; induction m <;> intros n h <;> simp
  case succ acc ih =>
    simp [pow_succ]; apply mul_ge_zero (ih _ h) h

theorem pow_gt_zero (n: Int) (m: Nat): n > 0 → n^m > 0 := by
  revert n; induction m <;> intros n h <;> simp
  case succ acc ih =>
    simp [pow_succ]; apply mul_gt_zero (ih _ h) h

theorem lt_add_lt_left (n m k: Int): n < m → k + n < k + m := by
  sorry

end Int

/-
### Properties on Nat
-/

theorem Nat.lt_of_add_lt_add_right {a b c: Nat} (h: a + c < b + c): a < b := by
  revert a b; induction c <;> intros a b h
  case zero =>
    exact h
  case succ c' ih =>
    apply lt_of_succ_lt_succ
    apply ih
    simp [Nat.succ_add, ←Nat.add_succ, h]

theorem Nat.minus_plus_one {a: Nat} (h: a > 0): a - 1 + 1 = a := by
  cases a; simp at h; rfl
