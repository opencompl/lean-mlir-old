import MLIR.Util.Arith
import MLIR.AST
import Mathlib.Tactic.LibrarySearch
open MLIR.AST (Signedness)

-- This file contains many sorrys for trivial arithmetic results that aren't
-- automated in Lean 4 yet (mainly because omega isn't ported). The following
-- tactic is used to dispatch them. We only use `sorry` for incomplete proofs.
macro "sorry_arith": tactic => `(tactic| sorry)

-- Stored as most significant bit first
inductive FinInt: Nat → Type :=
  | nil: FinInt 0
  | next: Bool → FinInt n → FinInt (n+1)
deriving DecidableEq

@[match_pattern]
abbrev FinInt.O: FinInt sz → FinInt (sz+1) := .next false
@[match_pattern]
abbrev FinInt.I: FinInt sz → FinInt (sz+1) := .next true

namespace FinInt

/-
### Computations modulo powers of 2
-/

-- Standard modulo: mod2 a n = a % 2^n

def mod2 (a: Int) (n: Nat): Int :=
  ((a % 2^n) + 2^n) % 2^n

theorem mod2_ge: mod2 a n ≥ 0 := by
  simp [mod2]
  apply Int.mod_ge
  have h := Int.ge_add_ge_right (2^n) (@Int.mod_ge_neg a (2^n))
  rw [Int.add_left_neg] at h
  assumption

theorem mod2_lt: mod2 a n < 2^n := by
  simp [mod2]
  apply Int.mod_lt Int.two_pow_pos

theorem mod2_bounds: mod2 a n ≥ 0 ∧ mod2 a n < 2^n :=
  ⟨mod2_ge, mod2_lt⟩

theorem mod2_idem {a: Int}: a ≥ 0 ∧ a < 2^n → mod2 a n = a := by
  intros h
  simp [mod2]
  rw [Int.add_mod_right Int.two_pow_pos (Int.mod_ge h.1), Int.mod_mod]
  apply Int.mod_bounds _ h.1 h.2

theorem mod2_idem_iff_bounds {a: Int}:
    mod2 a n = a ↔ (a ≥ 0 ∧ a < 2^n) :=
  ⟨(. ▸ mod2_bounds), mod2_idem⟩

@[simp]
theorem mod2_mod2: mod2 (mod2 a n) n = mod2 a n :=
  mod2_idem mod2_bounds

theorem mod2_fequal: x = y → mod2 x n = mod2 y n := by
  intros h; simp [h]

theorem mod2_zero: mod2 0 n = 0 := by
  simp [mod2]
  rw[Int.mod_eq_zero_of_dvd];
  simp[Int.dvd_refl];

theorem mod2_exp_n: mod2 (2^n) n = 0 := by
  simp [mod2]
  rw [Int.add_mod_right Int.two_pow_pos (Int.mod_ge Int.two_pow_ge)]
  simp [Int.mod_mod]
  simp [Int.mod_self]

theorem mod2_add_left: mod2 (2^n + x) n = mod2 x n := by
  sorry

theorem mod2_add_right: mod2 (x + 2^n) n = mod2 x n := by
  sorry

-- Symmetric modulo : smod2 a n = a % 2^(n+1) spread over -(2^n) ... 2^n-1

def smod2 (a: Int) (n: Nat): Int :=
  if mod2 a (n+1) ≥ 2^n then mod2 a (n+1) - 2^(n+1) else mod2 a (n+1)

theorem smod2_ge: smod2 a n ≥ -(2^n) := by
  simp [smod2]; split
  . sorry_arith
  . apply Int.ge_trans 0 mod2_ge (Int.zero_ge_neg Int.two_pow_ge)

theorem smod2_lt: smod2 a n < 2^n := by
  simp [smod2]; split
  . sorry_arith -- it's < 0 anyway
  . sorry_arith

theorem smod2_bounds: smod2 a n ≥ -(2^n) ∧ smod2 a n < 2^n :=
  ⟨smod2_ge, smod2_lt⟩

theorem smod2_idem {a: Int}: a ≥ -(2^n) ∧ a < 2^n → smod2 a n = a := by
  sorry

theorem smod2_idem_iff_bounds {a: Int}:
    smod2 a n = a ↔ (a ≥ -(2^n) ∧ a < 2^n) :=
  ⟨(. ▸ smod2_bounds), smod2_idem⟩

theorem smod2_smod2: smod2 (smod2 a n) n = smod2 a n :=
  smod2_idem smod2_bounds

-- Relation between modulo and symmetric modulo

theorem smod2_as_mod2: smod2 a n = mod2 (a + 2^n) (n+1) - 2^n := by
  sorry

-- Congruence

def cong2 (n: Nat): Int → Int → Prop :=
  fun a b => mod2 a n = mod2 b n

def scong2 (n: Nat): Int → Int → Prop :=
  fun a b => smod2 a n = smod2 b n

notation a " ≡ " b  " [2^" n "]" =>  cong2 n a b
notation a " ≡ " b " [±2^" n "]" => scong2 n a b

instance {n}: Equivalence (cong2 n) where
  refl _ := rfl
  symm := Eq.symm
  trans := Eq.trans

instance {n}: Equivalence (scong2 n) where
  refl _ := rfl
  symm := Eq.symm
  trans := Eq.trans

theorem cong2_to_eq (n: Nat):
    a ≡ b [2^n] →
    a ≥ 0 ∧ a < 2^n →
    b ≥ 0 ∧ b < 2^n →
    a = b := by
  intros h ha hb
  rw [←mod2_idem ha, ←mod2_idem hb]; assumption

theorem scong2_to_eq (n: Nat):
    a ≡ b [±2^n] →
    a ≥ -(2^n) ∧ a < 2^n →
    b ≥ -(2^n) ∧ b < 2^n →
    a = b := by
  intros h ha hb
  rw [←smod2_idem ha, ←smod2_idem hb]; assumption

@[simp]
theorem mod2_add_l: mod2 (mod2 a n + b) n = mod2 (a + b) n := by
  sorry

@[simp]
theorem mod2_add_r: mod2 (a + mod2 b n) n = mod2 (a + b) n := by
  sorry

@[simp]
theorem mod2_neg: mod2 (-mod2 a n) n = mod2 (-a) n := by
  sorry

@[simp]
theorem mod2_sub_r: mod2 (a - mod2 b n) n = mod2 (a - b) n := by
  sorry

@[simp]
theorem mod2_sub_l: mod2 (mod2 a n - b) n = mod2 (a - b) n := by
  sorry

theorem cong2_mod2_right: a ≡ b [2^n] → a ≡ mod2 b n [2^n] := by
  simp [cong2, mod2_idem mod2_bounds]

theorem cong2_mod2_left: a ≡ b [2^n] → mod2 a n ≡ b [2^n] := by
  simp [cong2, mod2_idem mod2_bounds]

/-
### Building FinInt from constants
-/

def zero: FinInt sz :=
  match sz with
  | 0 => .nil
  | sz+1 => .O zero

def one: FinInt sz :=
  match sz with
  | 0 => .nil
  | 1 => .I nil
  | sz+1 => .O one

def minusOne: FinInt sz :=
  match sz with
  | 0 => .nil
  | sz+1 => .I minusOne

instance: Inhabited (FinInt sz) where
  default := zero

def isInBounds (sgn: Signedness) (sz: Nat) (i: Int): Bool :=
  match sgn with
  | .Signless => i ≥ 0 ∧ i < 2^sz
  | .Unsigned => i ≥ 0 ∧ i < 2^sz
  | .Signed   => i ≥ -(2^(sz-1)) ∧ i < 2^(sz-1)

private def ofIntAux (sz: Nat) (n: Int): FinInt sz × Int :=
  match sz with
  | 0 => (.nil, n)
  | sz+1 =>
      match ofIntAux sz n with
      | (r, m) => (.next (m%2 == 1) r, m / 2)

def ofInt (sz: Nat) (i: Int): FinInt sz :=
  ofIntAux sz (mod2 i sz) |>.fst

instance {sz: Nat} {a: Nat}: OfNat (FinInt sz) a where
  ofNat := ofInt sz a

def ofIntAux_0 {sz: Nat}:
    ofIntAux sz 0 = (zero, 0) := by
  induction sz with
  | zero => decide
  | succ n h => simp [ofIntAux, h, zero]

def ofInt_0 {sz: Nat}:
    ofInt sz 0 = zero := by
  simp [ofInt, ofIntAux_0, mod2_zero]

def ofIntAux_1 {sz: Nat}:
    ofIntAux (sz+1) 1 = (one, 0) := by
  induction sz with
  | zero => decide
  | succ n h => rw [Nat.succ_add]; unfold ofIntAux; simp [h, one]

def ofInt_1 {sz: Nat}:
    ofInt sz 1 = one := by
  match sz with
  | 0 => decide
  | sz'+1 =>
    simp [ofInt, @mod2_idem (sz'+1) 1 ⟨by decide, by sorry_arith⟩]
    simp [ofIntAux_1]

/-
### Converting FinInt back to Int
-/

@[simp]
def toUint (n: FinInt sz): Int :=
  match sz, n with
  | 0,    .nil => 0
  | sz+1, .O i => i.toUint
  | sz+1, .I i => 2^sz + i.toUint

def toSint (n: FinInt sz): Int :=
  match sz, n with
  | 0, .nil => 0
  | sz+1, .O m => m.toUint
  | sz+1, .I m => m.toUint - 2^sz

-- | A variant of toSint that is defined at the singularity of
-- zero bit-width based on https://reviews.llvm.org/D116413
def toSint' (n: FinInt sz): Int :=
  match sz, n with
  | 0, .nil =>  0 -- is this sane?
  | sz+1, .O m => m.toUint
  | sz+1, .I m => m.toUint - 2^sz

theorem toUint_ge {n: FinInt sz}: n.toUint ≥ 0 := by
  revert n; induction sz <;> intros n <;> cases n <;> simp
  case next sz bn n' ih =>
    cases bn <;> simp [@ih n']
    apply Int.add_ge_zero; apply Int.pow_ge_zero; decide; apply ih

theorem toUint_lt {n: FinInt sz}: n.toUint < (2:Int)^sz := by
  revert n; induction sz <;> intros n <;> cases n <;> simp
  case succ.next sz bn n' ih =>
    cases bn <;> simp
    . apply Int.lt_trans 
      . specialize @ih n'; simp at ih; trivial
      . simp [Int.pow_succ, Int.mul_two]; apply Int.lt_add_right
        apply Int.pow_gt_zero; decide
    . specialize @ih n'; simp at ih; simp [Int.pow_succ, Int.mul_two]
      apply Int.lt_add_lt_left; trivial

theorem toUint_bounds {n: FinInt sz}:
    n.toUint ≥ 0 ∧ n.toUint < 2^sz :=
  ⟨toUint_ge, toUint_lt⟩

theorem toUint_mod2 {n: FinInt sz}: mod2 n.toUint sz = n.toUint :=
  mod2_idem toUint_bounds

theorem toSint_ge {n: FinInt (sz+1)}: n.toSint ≥ -(2^sz) := by
  cases n; case next bn n' =>
  cases bn <;> simp [toSint]
  . apply Int.le_trans  <;> sorry_arith
  . sorry_arith

theorem toSint_lt {n: FinInt (sz+1)}: n.toSint < 2^sz := by
  cases n; case next bn n' =>
  cases bn <;> simp [toSint]
  . apply toUint_lt
  . sorry_arith

theorem toSint_bounds {n: FinInt (sz+1)}:
    n.toSint ≥ -(2^sz) ∧ n.toSint < 2^sz :=
  ⟨toSint_ge, toSint_lt⟩

theorem toSint_smod2 {n: FinInt (sz+1)}: smod2 n.toSint sz = n.toSint :=
  smod2_idem toSint_bounds

theorem toUint_zero {sz}:
    toUint (@zero sz) = 0 := by
  induction sz with
  | zero => decide
  | succ n ih => simp [toUint, ih]

theorem toUint_one {sz}:
    sz > 0 → toUint (@one sz) = 1 := by
  intros h
  induction sz with
  | zero => cases (by decide: ¬ 0 > 0) h
  | succ n ih => cases n; decide; simp [toUint]; apply ih; simp_arith

theorem toUint_minusOne {sz}:
    toUint (@minusOne sz) = 2^sz - 1 := by
  induction sz with
  | zero => decide
  | succ n ih => simp [toUint, ih]; sorry_arith

/-
### Numerical bounds derived from bit structure
-/

theorem msb_bound (b: Bool) (n: FinInt sz):
    (FinInt.next b n).toUint < 2^sz ↔ b = false := by
  cases b <;> simp [toUint]
  . apply toUint_lt
  . apply Int.le_add_of_nonneg_right; simp[toUint_ge];


theorem O_lt (n: FinInt sz): (FinInt.next false n).toUint < 2^sz :=
  (msb_bound false n).2 rfl

theorem I_not_lt (n: FinInt sz): ¬(FinInt.next true n).toUint < 2^sz :=
  fun h => Bool.noConfusion <| (msb_bound true n).1 h

/-
### String representation
-/

def str_u (n: FinInt sz): String :=
  match sz, n with
  | 0,    .nil => ""
  | sz+1, .O m => "0 " ++ str_u m
  | sz+1, .I m => "1 " ++ str_u m

def str_s (n: FinInt (sz+1)): String :=
  match n with
  | .O m => "s=0 " ++ str_u m
  | .I m => "s=1 " ++ str_u m

-- The default is still to display them as unsigned integers
instance: ToString (FinInt sz) where
  toString n := s!"{n.toUint}[{sz}-bit]"

/-
### Conversions of sign/valuation and derived equalities
-/

theorem toSint_of_toUint (n: FinInt (sz+1)):
  n.toSint =
    match n with
    | .O _ => n.toUint
    | .I _ => n.toUint - 2^(sz+1) := by
  match n with
  | .O m => simp [toSint, toUint]
  | .I m => simp [toSint, toUint]; sorry_arith

theorem toUint_of_toSint (n: FinInt (sz+1)):
  n.toUint =
    match n with
    | .O _ => n.toSint
    | .I _ => n.toSint + 2^(sz+1) := by
  match n with
  | .O m => simp [toSint, toUint]
  | .I m => simp [toSint, toUint]; sorry_arith

theorem toUint_ofInt (sz: Nat) (n: Int):
    (ofInt sz n).toUint = mod2 n sz := by
  induction sz
  case zero =>
    simp [mod2]; sorry_arith
  case succ sz ih =>
    sorry

theorem toUint_ofNat (sz: Nat) (n: Nat):
    (OfNat.ofNat n: FinInt sz).toUint = mod2 n sz := by
  simp [OfNat.ofNat, toUint_ofInt]

theorem toSint_ofSint (sz: Nat)(n: Int):
    (ofInt (sz+1) n).toSint = smod2 n sz := by
  sorry

theorem ofIntAux_spec (sz: Nat) (n: Int):
    (FinInt.ofIntAux sz n).fst.toUint = mod2 n sz ∧
    (FinInt.ofIntAux sz n).snd = n / 2^sz := by
  induction sz
  case zero =>
    simp [FinInt.ofIntAux]; sorry_arith
  case succ sz ih =>
    simp [FinInt.ofIntAux, ih]; constructor
    . have h: (n / 2^sz % 2 = 0) ∨ (n / 2^sz %2 = 1) := by sorry
      match h with
      | .inl h => simp [h, ih]; sorry_arith
      | .inr h => simp [h, ih]; sorry_arith
    . sorry_arith

theorem eq_of_toUint_eq (a b: FinInt sz):
    a.toUint = b.toUint → a = b := by
  intros h <;> induction sz
  case zero =>
    cases a; cases b; rfl
  case succ sz ih =>
    cases a; case next sz da a' =>
    cases b; case next db b' =>
    cases da <;> cases db <;> simp [toUint] at h <;> simp
    . apply ih _ _ h
    . have h₁ := @toUint_lt _ a'; have h₂ := @toUint_ge _ b'; sorry_arith
    . have h₁ := @toUint_ge _ a'; have h₂ := @toUint_lt _ b'; sorry_arith
    . apply ih; sorry_arith

theorem eq_of_toUint_cong2 (a b: FinInt n):
    a.toUint ≡ b.toUint [2^n] → a = b := by
  intros h
  apply eq_of_toUint_eq
  apply cong2_to_eq n h toUint_bounds toUint_bounds

/-
### Logical operations
-/

def logic1 (f: Bool → Bool) (n: FinInt sz): FinInt sz :=
  match n with
  | .nil => .nil
  | .next bn n' => .next (f bn) (logic1 f n')

def comp: FinInt sz → FinInt sz :=
  logic1 (! ·)

def logic2 (f: Bool → Bool → Bool) (n m: FinInt sz): FinInt sz :=
  match n, m with
  | .nil, .nil => .nil
  | .next bn n', .next bm m' => .next (f bn bm) (logic2 f n' m')

def and: FinInt sz → FinInt sz → FinInt sz :=
  logic2 (· && ·)
def or: FinInt sz → FinInt sz → FinInt sz :=
  logic2 (· || ·)
def xor: FinInt sz → FinInt sz → FinInt sz :=
  logic2 (· != ·)

instance: HAnd (FinInt sz) (FinInt sz) (FinInt sz) where
  hAnd := FinInt.and

instance: HOr (FinInt sz) (FinInt sz) (FinInt sz) where
  hOr := FinInt.or

instance: HXor (FinInt sz) (FinInt sz) (FinInt sz) where
  hXor := FinInt.xor

theorem comp_toUint (n: FinInt sz):
    (comp n).toUint = 2^sz - 1 - n.toUint := by
  induction sz
  case zero =>
    cases n; simp
  case succ sz ih =>
    simp [comp] at ih
    cases n; case next sz bn n' =>
    cases bn <;> simp [ih]
    . sorry_arith
    . sorry_arith

/-
### Addition to n+1 bits
-/

def addfull (n m: FinInt sz): FinInt (sz+1) :=
  match sz, n, m with
  | 0, .nil, .nil => .O .nil
  | sz+1, .next bn n', .next bm m' =>
      match addfull n' m' with
      | .O r => .next (bn && bm) $ .next (bn != bm) r
      | .I r => .next (bn || bm) $ .next (bn == bm) r

protected theorem addfull_toUint (n m: FinInt sz):
    (addfull n m).toUint = n.toUint + m.toUint := by
  revert n m; induction sz <;> intros n m
  case zero =>
    cases n; cases m; simp [addfull]
  case succ sz ih =>
    cases n; case next sz dn n' =>
    cases m; case next dm m' =>
    simp [addfull]
    split
    case h_1 r h =>
      have h': r.toUint = n'.toUint + m'.toUint := by
        rw [←ih n' m']; simp [h]
      simp at h'
      cases dn <;> cases dm <;> simp [FinInt.toUint, h']
      <;> sorry_arith
    case h_2 r h =>
      have h': r.toUint = n'.toUint + m'.toUint - 2^sz := by
        rw [←ih n' m']
        simp [h, Int.add_comm, toSint]
      simp at h'
      cases dn <;> cases dm <;> simp [FinInt.toUint, h', toSint]
      <;> sorry_arith

theorem addfull_toSint (n m: FinInt (sz+1)):
  (addfull n m).toSint =
    match n, m with
    | .O n', .O m' =>
        n.toSint + m.toSint -- carry is always 0
    | .O n', .I m' =>
        match addfull n m with
        | .O _ => n.toSint + m.toSint + 2^(sz+1)
        | .I _ => n.toSint + m.toSint - 2^(sz+1)
    | .I n', .O m' =>
        match addfull n m with
        | .O _ => n.toSint + m.toSint + 2^(sz+1)
        | .I _ => n.toSint + m.toSint - 2^(sz+1)
    | .I n', .I m' =>
        n.toSint + m.toSint := by -- carry is always 1
  cases n; case next bn n' =>
  cases m; case next bm m' =>
  cases bn <;> cases bm <;> simp [addfull]
  case false.false =>
    match h: addfull n' m' with
    | .next br r =>
      cases br <;> simp [toSint, ←FinInt.addfull_toUint, h]
  case false.true =>
    match h: addfull n' m' with
    | .next br r =>
      have h' := FinInt.addfull_toUint n' m'
      rw [h] at h'
      cases br <;> simp at h' <;> simp [toSint]
      . simp [h']; sorry_arith
      . sorry_arith
  case true.false =>
    match h: addfull n' m' with
    | .next br r =>
      have h' := FinInt.addfull_toUint n' m'
      rw [h] at h'
      cases br <;> simp at h' <;> simp [toSint]
      . simp [h']; sorry_arith
      . sorry_arith
  case true.true =>
    match h: addfull n' m' with
    | .next br r =>
      have h' := FinInt.addfull_toUint n' m'
      rw [h] at h'
      cases br <;> simp at h' <;> simp [toSint, h'] <;> sorry_arith

theorem addfull_comm (n m: FinInt sz):
    addfull n m = addfull m n := by
  induction sz <;> cases n <;> cases m <;> simp [addfull]
  case succ.next.next sz bn n' ih bm m' =>
    simp [ih n' m']
    split <;> cases bn <;> cases bm <;> rfl

/-
### Addition to n bits
-/

def addc (n m: FinInt sz): FinInt sz × Bool :=
  match addfull n m with
  | .next carry n => (n, carry)

theorem addc_comm (n m: FinInt sz): addc n m = addc m n := by
  simp [addc, addfull_comm]

def add (n m: FinInt sz): FinInt sz :=
  match addfull n m with
  | .next carry n => n

theorem add_comm' (n m: FinInt sz): add n m = add m n := by
  simp [add, addfull_comm]

instance {sz}: HAdd (FinInt sz) (FinInt sz) (FinInt sz) where
  hAdd := add

theorem add_comm (n m: FinInt sz): n + m = m + n := by
  simp [HAdd.hAdd, add_comm']

protected theorem add_toUint_msb (n m: FinInt sz):
  (n + m).toUint =
    match addfull n m with
    | .O _ => n.toUint + m.toUint
    | .I _ => n.toUint + m.toUint - 2^sz := by
  simp [add]
  match h: addfull n m with
  | .next br r =>
    have h₁ := FinInt.addfull_toUint n m
    rw [h] at h₁
    have h₂: n+m = r := by simp [HAdd.hAdd, add, h]
    cases br <;> simp [h₂] at *
    . exact h₁
    . sorry_arith

theorem add_toUint_by_lt (n m: FinInt sz):
    n.toUint + m.toUint < 2^sz →
    (n + m).toUint = n.toUint + m.toUint := by
  intros h_bound
  simp [HAdd.hAdd, FinInt.add, FinInt.addfull_toUint]
  cases h₁: addfull n m
  case next carry r =>
    have h₂: (addfull n m).toUint = (next carry r).toUint := by simp [h₁]
    cases carry <;> simp at h₂
    . simp [←h₂, FinInt.addfull_toUint, HAdd.hAdd]
    . simp [FinInt.addfull_toUint] at h₂
      rw [h₂] at h_bound
      sorry_arith -- contradiction at h_bound

theorem add_toUint_rem (n m: FinInt sz):
    (n + m).toUint =
      if n.toUint + m.toUint < 2^sz then
        n.toUint + m.toUint
      else
        n.toUint + m.toUint - 2^sz := by
  simp [FinInt.add_toUint_msb]
  cases h_full: addfull n m
  case next br r' =>
  cases br <;> simp
  . have h := O_lt r'; rw [←h_full] at h
    simp [FinInt.addfull_toUint] at h; simp [h]
  . have h := I_not_lt r';
    rw [←h_full] at h
    rw [FinInt.addfull_toUint] at h;
    simp [h]

theorem add_toUint (n m: FinInt sz):
    (n + m).toUint = mod2 (n.toUint + m.toUint) sz := by
  rw [add_toUint_rem]; split
  . simp [mod2_idem ⟨Int.add_ge_zero _ _ toUint_ge toUint_ge, by assumption⟩]
  . sorry_arith

theorem add_toUint_cong2 (n m: FinInt sz):
    (n + m).toUint ≡ n.toUint + m.toUint [2^sz] := by
  simp [cong2, toUint_mod2]; apply add_toUint

theorem add_toSint_msb (n m: FinInt (sz+1)):
  (n + m).toSint =
    match n, m with
    | .O n', .O m' =>
        match n + m with
        | .O _ => n.toSint + m.toSint
        | .I _ => n.toSint + m.toSint - 2^(sz+1)
    | .O n', .I m' =>
        n.toSint + m.toSint
    | .I n', .O m' =>
        n.toSint + m.toSint
    | .I n', .I m' =>
        match n + m with
        | .O _ => n.toSint + m.toSint + 2^(sz+1)
        | .I _ => n.toSint + m.toSint := by
  simp [add]
  have h := addfull_toSint n m
  cases n; case next bn n' =>
  cases m; case next bm m' =>
  cases bn <;> cases bm <;> simp [addfull] at *
  case false.false =>
    match h': addfull n' m' with
    | .next br r =>
      rw [h'] at h
      cases br <;> simp [bne, toSint] at * <;> sorry_arith
  case false.true =>
    match h': addfull n' m' with
    | .next br r =>
      rw [h'] at h
      cases br <;> simp [bne, BEq.beq, toSint] at * <;> sorry_arith
  case true.false =>
    match h': addfull n' m' with
    | .next br r =>
      rw [h'] at h
      cases br <;> simp [bne, BEq.beq, toSint] at * <;> sorry_arith
  case true.true =>
    match h': addfull n' m' with
    | .next br r =>
      rw [h'] at h
      cases br <;> simp [bne, toSint] at * <;> sorry_arith

def addv (n m: FinInt (sz+1)): FinInt (sz+1) × Bool :=
  match n, m with
  | .O n', .O m' =>
    match n + m with
    | .next msb r' => (.next msb r', msb)
  | .I n', .I m' =>
    match n + m with
    | .next msb r' => (.next msb r', !msb)
  | _, _ =>
      (n + m, false)

theorem addv_comm (n m: FinInt (sz+1)): addv n m = addv m n := by
  simp [addv]; cases n; cases m; case next.next bn n' bm m' =>
  simp [add_comm]; cases bn <;> cases bm <;> rfl

theorem addv_toSint (n m: FinInt (sz+1)):
    (addv n m).snd = false →
    (addv n m).fst.toSint = n.toSint + m.toSint := by
  intros h_nov
  simp [addv] at *
  have h := add_toSint_msb n m
  cases n; case next dn n' =>
  cases m; case next dm m' =>
  cases dn <;> cases dm <;> simp [h] at *
  case false.false =>
    match h: next false n' + next false m' with
    | .next br r =>
      rw [h] at h_nov
      cases br <;> simp at *
      rw [←h]; simp [add_toSint_msb]; simp [h]
  case true.true =>
    match h: next true n' + next true m' with
    | .next br r =>
      rw [h] at h_nov
      cases br <;> simp at *
      rw [←h]; simp [add_toSint_msb]; simp [h]

/-
### Negation operator
-/

def neg (n: FinInt sz): FinInt sz :=
  comp n + 1

instance: Neg (FinInt sz) where
  neg := neg

theorem neg_toUint (n: FinInt sz):
    (-n).toUint = mod2 (-n.toUint) sz := by
  sorry

theorem neg_toUint_cong2 (n: FinInt sz):
    (-n).toUint ≡ -n.toUint [2^sz] := by
  simp [cong2, toUint_mod2]; apply neg_toUint

theorem neg_minusOne {sz: Nat}:
    (-1: FinInt sz) = minusOne := by
  match H: sz with
  | 0 => decide
  | sz'+1 =>
      have h: sz' + 1 > 0 := by sorry_arith
      apply eq_of_toUint_cong2
      simp [toUint_minusOne]
      simp [neg_toUint]
      apply cong2_mod2_left
      simp [ofInt_1, toUint_one h]
      sorry_arith

theorem comp_eq_xor_minusOne (n: FinInt sz):
    comp n = n ^^^ -1 := by
  simp [neg_minusOne, HXor.hXor]
  induction sz <;> simp [comp, xor, logic1, logic2, minusOne] at *
  case zero =>
    cases n; simp
  case succ sz ih =>
    match n with
    | .I m => simp [logic1, logic2, ih]
    | .O m => simp [logic1, logic2, ih]

/-
### Subtraction
-/

def sub (n m: FinInt sz): FinInt sz :=
  n + -m

instance: Sub (FinInt sz) where
  sub := sub

theorem sub_toUint (n m: FinInt sz):
    (n - m).toUint = mod2 (n.toUint - m.toUint) sz := by
  sorry

theorem sub_toUint_cong2 (n: FinInt sz):
    (n - m).toUint ≡ n.toUint - m.toUint [2^sz] := by
  simp [cong2, toUint_mod2]; apply sub_toUint

/-
### Sign extensions
-/

-- TODO: The MSB first encoding makes it really hard to do zext structurally
-- without type casts
def zext {sz₁: Nat} (sz₂: Nat) (n: FinInt sz₁): FinInt sz₂ :=
  .ofInt sz₂ n.toUint

theorem zext_toUint: (@zext sz₁ sz₂ n).toUint = mod2 n.toUint sz₂ := by
  simp [zext, toUint_ofInt]

theorem zext_toUint': sz₁ < sz₂ → (@zext sz₁ sz₂ n).toUint = n.toUint := by
  simp [zext, toUint_ofInt]
  intros h
  apply mod2_idem ⟨toUint_ge, _⟩
  apply Int.lt_trans toUint_lt
  exact (sorry: 2^sz₁ < 2^sz₂)

/-
### Select operation
-/

def select {α: Type} (b: FinInt 1) (a₁ a₂: α): α :=
  if b.toUint = 1 then a₁ else a₂

theorem bool_cases (b: FinInt 1):
    b = .ofInt _ 0 ∨ b = .ofInt _ 1 :=
  match b with
  | .O .nil => .inl rfl
  | .I .nil => .inr rfl

end FinInt
