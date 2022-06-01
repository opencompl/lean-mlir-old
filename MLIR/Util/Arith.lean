-- Int

theorem Int.n_sub_0 (n: Int): n - 0 = n := by
  cases n <;> rfl

theorem Int.ge_0_NonNeg (n: Int): n ≥ 0 ↔ Int.NonNeg n := by
  simp [GE.ge, LE.le, Int.le]
  simp [Int.n_sub_0]

theorem Int.mul_ge_0 (n m: Int): n ≥ 0 → m ≥ 0 → n * m ≥ 0 := by
  rw [Int.ge_0_NonNeg, Int.ge_0_NonNeg, Int.ge_0_NonNeg]
  intros hn hm;
  cases hn; cases hm; constructor

theorem Int.mul_comm (n m: Int): n * m = m * n := by
  sorry

theorem Int.mul_add (n m k: Int): n * (m + k) = n * m + n * k := by
  sorry

@[simp]
theorem Int.add_assoc (n m k: Int): n + (m + k) = n + m + k := by
  sorry

theorem Int.add_comm (n m: Int): n + m = m + n := by
  sorry

-- simp_arith generates a term that doesn't type?
@[simp]
theorem Int.mul_zero (n: Int): n * 0 = 0 := by
  simp [HMul.hMul, Mul.mul, Int.mul]
  cases n <;> simp

@[simp]
theorem Int.add_zero (n: Int): n + 0 = n := by
  simp [HAdd.hAdd, Add.add, Int.add]
  cases n <;> simp
  sorry

-- Nat

theorem Nat.lt_of_add_lt_add_right {a b c: Nat} (h: a + c < b + c):
    a < b := by
  sorry

theorem Nat.minus_plus_one {a: Nat} (h: a > 0):
    a - 1 + 1 = a := by
  sorry
