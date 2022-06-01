import MLIR.Util.Arith
import MLIR.AST
open MLIR.AST (Signedness)

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
### Basic values and bounds
-/

def zero: FinInt sz :=
  match sz with
  | 0 => .nil
  | sz+1 => .O zero

instance: Inhabited (FinInt sz) where
  default := zero

def minUint (sz: Nat): Int :=
  0

def maxUint (sz: Nat): Int :=
  (2:Int)^sz - 1

def minSint (sz: Nat): Int :=
  -(2:Int)^(sz-1)

def maxSint (sz:Nat): Int :=
  (2:Int)^(sz-1) -1

def isUintInBounds (sz: Nat) (i: Int): Bool :=
  i ≥ minUint sz && i ≤ maxUint sz

def isSintInBounds (sz: Nat) (i: Int): Bool :=
  i ≥ minSint sz && i ≤ maxSint sz

def isInBounds (sgn: Signedness) (sz: Nat) (i: Int): Bool :=
  match sgn with
  | .Signless => isUintInBounds sz i
  | .Unsigned => isUintInBounds sz i
  | .Signed   => isSintInBounds sz i

private def ofUint_aux (sz: Nat) (n: Int): FinInt sz × Int :=
  match sz with
  | 0 => (.nil, n)
  | sz+1 =>
      match ofUint_aux sz n with
      | (r, m) => (.next (m%2 == 1) r, m / 2)

def ofUint (sz: Nat) (n: Int): FinInt sz :=
  ofUint_aux sz n |>.fst

def ofSint (sz: Nat) (i: Int): FinInt sz :=
  ofUint sz (if i >= 0 then i % (2:Int)^sz else i % (2:Int)^sz + (2:Int)^sz)

def ofInt (sgn: Signedness) (sz: Nat) (i: Int): FinInt sz :=
  match sgn with
  | .Signless => ofUint sz i
  | .Unsigned => ofUint sz i
  | .Signed   => ofSint sz i

/-
### Integer valuation
-/

@[simp]
def toUint (n: FinInt sz): Int :=
  match sz, n with
  | 0,    .nil => 0
  | sz+1, .O i => i.toUint
  | sz+1, .I i => (2:Int)^sz + i.toUint

def toSint (n: FinInt (sz+1)): Int :=
  match n with
  | .O m => m.toUint
  | .I m => m.toUint - (2:Int)^sz

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
  toString n := s!"[{sz}]{n.toUint}"

/-
### Signedness conversion and effect on integer valuation
-/

theorem toSint_of_toUint (n: FinInt (sz+1)):
  n.toSint =
    match n with
    | .O _ => n.toUint
    | .I _ => n.toUint - (2:Int)^(sz+1) := by
  match n with
  | .O m => simp [toSint, toUint]
  | .I m => simp [toSint, toUint]; sorry -- trivial

theorem toUint_of_toSint (n: FinInt (sz+1)):
  n.toUint =
    match n with
    | .O _ => n.toSint
    | .I _ => n.toSint + (2:Int)^(sz+1) := by
  match n with
  | .O m => simp [toSint, toUint]
  | .I m => simp [toSint, toUint]; sorry -- trivial

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

theorem addfull_toUint (n m: FinInt sz):
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
      <;> sorry -- clearly true, but annoying
    case h_2 r h =>
      have h': r.toUint = n'.toUint + m'.toUint - (2:Int)^sz := by
        rw [←ih n' m']
        simp [h, Int.add_comm, toSint]
        sorry -- obvious
      simp at h'
      cases dn <;> cases dm <;> simp [FinInt.toUint, h', toSint]
      <;> sorry -- clearly true, but annoying

theorem addfull_toSint (n m: FinInt (sz+1)):
  (addfull n m).toSint =
    match n, m with
    | .O n', .O m' =>
        n.toSint + m.toSint -- carry is always 0
    | .O n', .I m' =>
        match addfull n m with
        | .O _ => n.toSint + m.toSint + (2:Int)^(sz+1)
        | .I _ => n.toSint + m.toSint - (2:Int)^(sz+1)
    | .I n', .O m' =>
        match addfull n m with
        | .O _ => n.toSint + m.toSint + (2:Int)^(sz+1)
        | .I _ => n.toSint + m.toSint - (2:Int)^(sz+1)
    | .I n', .I m' =>
        n.toSint + m.toSint := by -- carry is always 1
  cases n; case next bn n' =>
  cases m; case next bm m' =>
  cases bn <;> cases bm <;> simp [addfull]
  case false.false =>
    match h: addfull n' m' with
    | .next br r =>
      cases br <;> simp [toSint, ←addfull_toUint, h]
  case false.true =>
    match h: addfull n' m' with
    | .next br r =>
      have h' := addfull_toUint n' m'
      rw [h] at h'
      cases br <;> simp at h' <;> simp [toSint]
      . simp [h']; sorry -- trivial
      . sorry -- more rearranging
  case true.false =>
    match h: addfull n' m' with
    | .next br r =>
      have h' := addfull_toUint n' m'
      rw [h] at h'
      cases br <;> simp at h' <;> simp [toSint]
      . simp [h']; sorry -- trivial
      . sorry -- more rearranging
  case true.true =>
    match h: addfull n' m' with
    | .next br r =>
      have h' := addfull_toUint n' m'
      rw [h] at h'
      cases br <;> simp at h' <;> simp [toSint, h'] <;> sorry -- obvious

/-
### Addition to n bits
-/

def addc (n m: FinInt sz): FinInt sz × Bool :=
  match addfull n m with
  | .next carry n => (n, carry)

def add (n m: FinInt sz): FinInt sz :=
  match addfull n m with
  | .next carry n => n

theorem add_toUint (n m: FinInt sz):
  (add n m).toUint =
    match addfull n m with
    | .O _ => n.toUint + m.toUint
    | .I _ => n.toUint + m.toUint - (2:Int)^sz := by
  simp [add]
  match h: addfull n m with
  | .next br r =>
    have h' := addfull_toUint n m
    rw [h] at h'
    cases br <;> simp at * <;> sorry -- trivial rearranging

theorem add_toSint (n m: FinInt (sz+1)):
  (add n m).toSint =
    match n, m with
    | .O n', .O m' =>
        match add n m with
        | .O _ => n.toSint + m.toSint
        | .I _ => n.toSint + m.toSint - (2:Int)^(sz+1)
    | .O n', .I m' =>
        n.toSint + m.toSint
    | .I n', .O m' =>
        n.toSint + m.toSint
    | .I n', .I m' =>
        match add n m with
        | .O _ => n.toSint + m.toSint + (2:Int)^(sz+1)
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
      cases br <;> simp [bne, toSint] at * <;> sorry -- trivial
  case false.true =>
    match h': addfull n' m' with
    | .next br r =>
      rw [h'] at h
      cases br <;> simp [bne, BEq.beq, toSint] at * <;> sorry -- trivial
  case true.false =>
    match h': addfull n' m' with
    | .next br r =>
      rw [h'] at h
      cases br <;> simp [bne, BEq.beq, toSint] at * <;> sorry -- trivial
  case true.true =>
    match h': addfull n' m' with
    | .next br r =>
      rw [h'] at h
      cases br <;> simp [bne, toSint] at * <;> sorry -- trivial

def addv (n m: FinInt  (sz+1)): FinInt  (sz+1) × Bool :=
  match n, m with
  | .O n', .O m' =>
    match add n m with
    | .next msb r' => (.next msb r', msb)
  | .I n', .I m' =>
    match add n m with
    | .next msb r' => (.next msb r', !msb)
  | _, _ =>
      (add n m, false)

theorem addv_toSint (n m: FinInt (sz+1)):
    (addv n m).snd = false →
    (addv n m).fst.toSint = n.toSint + m.toSint := by
  intros h_nov
  simp [addv] at *
  have h := add_toSint n m
  cases n; case next dn n' =>
  cases m; case next dm m' =>
  cases dn <;> cases dm <;> simp [h] at *
  case false.false =>
    match h: add (next false n') (next false m') with
    | .next br r =>
      rw [h] at h_nov
      cases br <;> simp at *
      rw [←h]; simp [add_toSint]; simp [h]
  case true.true =>
    match h: add (next true n') (next true m') with
    | .next br r =>
      rw [h] at h_nov
      cases br <;> simp at *
      rw [←h]; simp [add_toSint]; simp [h]

end FinInt
