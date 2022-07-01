import MLIR.Util.Arith
import MLIR.AST
open MLIR.AST (Signedness)

-- This file contains many sorrys for trivial arithmetic results that aren't
-- automated in Lean 4 yet (mainly because omega isn't ported). The following
-- tactic is used to dispatch them. We only use `sorry` for incomplete proofs.
macro "sorry_arith": tactic => `(tactic| sorry)

-- Write 2^n in Int contexts instead of (2:Int)^n
instance: HPow Nat Nat Int where
  hPow x y := (x:Int)^y

-- Stored as most significant bit first
inductive FinInt: Nat → Type :=
  | nil: FinInt 0
  | next: Bool → FinInt n → FinInt (n+1)
deriving DecidableEq

@[matchPattern]
abbrev FinInt.O: FinInt sz → FinInt (sz+1) := .next false
@[matchPattern]
abbrev FinInt.I: FinInt sz → FinInt (sz+1) := .next true

namespace FinInt

/-
### Computations modulo powers of 2
-/

-- Standard modulo: mod2 a n = a % 2^n

def mod2 (a: Int) (n: Nat): Int :=
  match a with
  | .ofNat p => .ofNat (p % 2^n)
  | .negSucc p => .negSucc (p % 2^n) + 2^n

theorem mod2_ge: mod2 a n ≥ 0 := by
  cases a <;> simp [mod2]
  . simp [Int.ge_zero_eq_nonneg]; constructor
  . sorry_arith

theorem mod2_lt: mod2 a n < 2^n := by
  cases a <;> simp [mod2]
  . sorry_arith
  . sorry_arith

theorem mod2_bounds: mod2 a n ≥ 0 ∧ mod2 a n < 2^n :=
  ⟨mod2_ge, mod2_lt⟩

theorem mod2_idem {a: Int}: a ≥ 0 ∧ a < 2^n → mod2 a n = a := by
  intros h
  cases a <;> simp [mod2]
  . sorry_arith
  . sorry_arith

theorem mod2_idem_iff_bounds {a: Int}:
    mod2 a n = a ↔ (a ≥ 0 ∧ a < 2^n) :=
  ⟨(. ▸ mod2_bounds), mod2_idem⟩

@[simp]
theorem mod2_mod2: mod2 (mod2 a n) n = mod2 a n :=
  mod2_idem mod2_bounds

theorem mod2_zero: mod2 (2^n) n = 0 := by
  sorry

-- Symmetric modulo : smod2 a n = a % 2^(n+1) spread over -2^n ... 2^n-1

def smod2 (a: Int) (n: Nat): Int :=
  if mod2 a (n+1) ≥ 2^n then mod2 a (n+1) - 2^(n+1) else mod2 a (n+1)

theorem smod2_ge: smod2 a n ≥ -2^n := by
  simp [smod2]; split
  . sorry_arith
  . sorry_arith -- it's ≥ 0 anyway

theorem smod2_lt: smod2 a n < 2^n := by
  simp [smod2]; split
  . sorry_arith -- it's < 0 anyway
  . sorry_arith

theorem smod2_bounds: smod2 a n ≥ -2^n ∧ smod2 a n < 2^n :=
  ⟨smod2_ge, smod2_lt⟩

theorem smod2_idem {a: Int}: a ≥ -2^n ∧ a < 2^n → smod2 a n = a := by
  sorry

theorem smod2_idem_iff_bounds {a: Int}:
    smod2 a n = a ↔ (a ≥ -2^n ∧ a < 2^n) :=
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

macro a:term " ≡ " b:term "[2^" n:term "]": term => `(cong2 $n $a $b)

macro a:term " ≡ " b:term "[±2^" n:term "]": term => `(scong2 $n $a $b)

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
    a ≥ -2^n ∧ a < 2^n →
    b ≥ -2^n ∧ b < 2^n →
    a = b := by
  intros h ha hb
  rw [←smod2_idem ha, ←smod2_idem hb]; assumption

theorem mod2_add: mod2 (a+b) n ≡ mod2 a n + mod2 b n [2^n] := by
  sorry

theorem mod2_mul: mod2 (a*b) n ≡ mod2 a n * mod2 b n [2^n] := by
  sorry

theorem mod2_neg: mod2 (-a) n ≡ -mod2 a n [2^n] := by
  sorry

@[simp]
theorem mod2_sub_r: mod2 (a - mod2 b n) n = mod2 (a - b) n := by
  sorry

@[simp]
theorem mod2_sub_l: mod2 (mod2 a n - b) n = mod2 (a - b) n := by
  sorry

theorem cong2_mod2_right: a ≡ b [2^n] → a ≡ mod2 b n [2^n] := by
  simp [cong2, mod2_idem mod2_bounds]; exact id

theorem cong2_mod2_left: a ≡ b [2^n] → mod2 a n ≡ b [2^n] := by
  simp [cong2, mod2_idem mod2_bounds]; exact id

/-
### Building FinInt from constants
-/

def zero: FinInt sz :=
  match sz with
  | 0 => .nil
  | sz+1 => .O zero

instance: Inhabited (FinInt sz) where
  default := zero

def isInBounds (sgn: Signedness) (sz: Nat) (i: Int): Bool :=
  match sgn with
  | .Signless => i ≥ 0 ∧ i < 2^sz
  | .Unsigned => i ≥ 0 ∧ i < 2^sz
  | .Signed   => i ≥ -2^(sz-1) ∧ i < 2^(sz-1)

private def ofUint_aux (sz: Nat) (n: Int): FinInt sz × Int :=
  match sz with
  | 0 => (.nil, n)
  | sz+1 =>
      match ofUint_aux sz n with
      | (r, m) => (.next (m%2 == 1) r, m / 2)

def ofUint (sz: Nat) (n: Int): FinInt sz :=
  ofUint_aux sz n |>.fst

def ofSint (sz: Nat) (i: Int): FinInt sz :=
  ofUint sz (if i >= 0 then i % 2^sz else i % 2^sz + 2^sz)

def ofInt (sgn: Signedness) (sz: Nat) (i: Int): FinInt sz :=
  match sgn with
  | .Signless => ofUint sz i
  | .Unsigned => ofUint sz i
  | .Signed   => ofSint sz i

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
    . apply Int.lt_trans ((2:Int)^sz)
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

theorem toSint_ge {n: FinInt (sz+1)}: n.toSint ≥ -(2:Int)^sz := by
  cases n; case next bn n' =>
  cases bn <;> simp [toSint]
  . apply Int.le_trans 0 <;> sorry_arith
  . sorry_arith

theorem toSint_lt {n: FinInt (sz+1)}: n.toSint < (2:Int)^sz := by
  cases n; case next bn n' =>
  cases bn <;> simp [toSint]
  . apply toUint_lt
  . sorry_arith

theorem toSint_bounds {n: FinInt (sz+1)}:
    n.toSint ≥ -2^sz ∧ n.toSint < 2^sz :=
  ⟨toSint_ge, toSint_lt⟩

theorem toSint_smod2 {n: FinInt (sz+1)}: smod2 n.toSint sz = n.toSint :=
  smod2_idem toSint_bounds

/-
### Numerical bounds derived from bit structure
-/

theorem msb_bound (b: Bool) (n: FinInt sz):
    (FinInt.next b n).toUint < 2^sz ↔ b = false := by
  cases b <;> simp [toUint]
  . apply toUint_lt
  . intro h; sorry_arith

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

theorem toUint_ofUint (sz: Nat) (n: Int):
    (ofUint sz n).toUint = mod2 n sz := by
  induction sz
  case zero =>
    simp [mod2]; sorry_arith
  case succ sz ih =>
    sorry

theorem toSint_ofSint (sz: Nat)(n: Int):
    (ofSint (sz+1) n).toSint = smod2 n sz := by
  sorry

theorem ofUint_aux_spec (sz: Nat) (n: Int):
    (FinInt.ofUint_aux sz n).fst.toUint = mod2 n sz ∧
    (FinInt.ofUint_aux sz n).snd = n / 2^sz := by
  induction sz
  case zero =>
    simp [FinInt.ofUint_aux]; sorry_arith
  case succ sz ih =>
    simp [FinInt.ofUint_aux, ih]; constructor
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

/-
### Integer valuation of `comp`
-/

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
        sorry_arith
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
  . have h := I_not_lt r'; rw [←h_full] at h
    simp [FinInt.addfull_toUint] at h; simp [h]

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
  comp n + .ofInt .Signless sz 1

instance: Neg (FinInt sz) where
  neg := neg

theorem neg_toUint (n: FinInt sz):
    (-n).toUint = mod2 (-n.toUint) sz := by
  sorry

theorem neg_toUint_cong2 (n: FinInt sz):
    (-n).toUint ≡ -n.toUint [2^sz] := by
  simp [cong2, toUint_mod2]; apply neg_toUint

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

end FinInt
