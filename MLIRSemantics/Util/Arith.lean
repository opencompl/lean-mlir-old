-- set_option pp.all true

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