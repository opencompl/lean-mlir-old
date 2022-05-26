/-
## Pattern unification

This file implements a unification algorithm for the matching patterns defined
by the framework. This is mainly used with PDL rewrites, where we unify the
rewrite's matching pattern with the dialect-provided matching patterns for all
the instructions targeted by the rewrite, in order to propagate operations
invariants that are implicit in PDL.
-/

import MLIR.Semantics.Types
import MLIR.Semantics.Matching
import MLIR.AST
open MLIR.AST

/-
### Unification equalities

These are simply typed equalities which make up the core of a unification
problem.

TODO: We must also have a unification equality for MLIR attributes.
-/

inductive UEq :=
  | EqValue: MValue → MValue → UEq
  | EqType: MType → MType → UEq
  | EqOp: MOp → MOp → UEq

-- Common instances

deriving instance Inhabited for UEq

def UEq.str: UEq → String
  | EqValue left right
  | EqType left right
  | EqOp left right =>
      s!"{left} ≡ {right}"

instance: ToString UEq := ⟨UEq.str⟩

-- Extensions of functions on matching patterns

def UEq.valueVars: UEq → List String
  | EqValue v₁ v₂ => v₁.valueVars ++ v₂.valueVars
  | EqType t₁ t₂  => []
  | EqOp op₁ op₂  => op₁.valueVars ++ op₂.valueVars

def UEq.typeVars: UEq → List String
  | EqValue v₁ v₂ => []
  | EqType t₁ t₂  => t₁.typeVars ++ t₂.typeVars
  | EqOp op₁ op₂  => op₁.typeVars ++ op₂.typeVars

def UEq.occurs (name: String): UEq → Bool
  | EqValue v₁ v₂ => v₁.occurs name || v₂.occurs name
  | EqType t₁ t₂  => t₁.occurs name || t₂.occurs name
  | EqOp op₁ op₂  => op₁.occurs name || op₂.occurs name

def UEq.substValue (eq: UEq) (name: String) (repl: MValue): UEq :=
  match eq with
  | EqValue v₁ v₂ => EqValue (v₁.substValue name repl) (v₂.substValue name repl)
  | EqType t₁ t₂  => EqType t₁ t₂
  | EqOp op₁ op₂  => EqOp (op₁.substValue name repl) (op₂.substValue name repl)

def UEq.substType (eq: UEq) (name: String) (repl: MType): UEq :=
  match eq with
  | EqValue v₁ v₂ => EqValue v₁ v₂
  | EqType t₁ t₂  => EqType (t₁.substType name repl) (t₂.substType name repl)
  | EqOp op₁ op₂  => EqOp (op₁.substType name repl) (op₂.substType name repl)

/-
### Unification problem

The Unification structure, representing a whole unification problem, tracks a
number of elements related to the resolution, including the original set of
equations and the substitutions that form the most general unifier.
-/

structure Unification where mk ::
  -- Original set of equations
  initial_equations: List UEq := []
  -- Current set of equations
  equations: List UEq := []
  -- Substitutions performed so far. These are restricted to have variables as
  -- the left hand of every equality.
  -- TODO: More specific definition of Unification.subst?
  substs: List UEq := []

-- Common instances

deriving instance Inhabited for Unification

def Unification.empty: Unification :=
  default

def Unification.str: Unification → String := fun u =>
  if u.equations.isEmpty then
    "(empty unification problem: no equations)"
  else
    "\n".intercalate (u.equations.map toString)

instance: ToString Unification := ⟨Unification.str⟩

def Unification.repr: Unification → String := fun u =>
  "Initial equations:\n" ++
    (String.join $ u.initial_equations.map (s!"  {·}\n")) ++
  "Current equations:\n" ++
    (String.join $ u.equations.map (s!"  {·}\n")) ++
  "Substitutions so far:\n" ++
    (String.join $ u.substs.map (s!"  {·}\n"))

/-
### Unification algorithm

We use a naive unification algorithm. Given the size of the patterns at play,
performance is not a real concern, and instead we prefer to have a more natural
traversal of the structure that leads to more intuitive logs and error
messages.

This is algorithm 1 of [1], which essentially normalizes the set of equations
through a number of unifier-preserving transformations.

For maintenance reasons, we log the progress of the algorithm as we go. Most of
the functions return IO (_ × Bool) or IO (Option (_ × Bool)); IO is obviously
for logging; the Bool is to indicate whether progress was made; and the Option
is set to none when the problem is found to be unsatisfiable.

[1] Martelli, Alberto, and Ugo Montanari. "An efficient unification algorithm."
    ACM Transactions on Programming Languages and Systems (TOPLAS) 4.2 (1982):
    258-282.

### Algorihm description

We run the rules below till fixpoint:
  1. Remove x = x.
  2. If we have `x = term`, and there are no cycles, we substitute.
  3. If we have `term = x`, we re-orient to `x = term`.
  4. If we have `constructor(a1, ... ak) = constructor(b1, ... bk), we add equalities `ai = bi`.
  5. Make sure that we have substitution priority right.
-/

-- ORIENT: turn [t = x] (where t not a variable) into [x = t]
-- We also orient equations [y = x] if y is a variable with a higher
-- substitution priority than x.

open UEq MValue MType MOp

private def orientOne (eqn: UEq): IO (UEq × Bool) :=
  match eqn with
  -- Orient to substitute automatically generated names first
  | EqValue (ValueVar p₁ n₁) (ValueVar p₂ n₂) =>
      if p₂ > p₁ then do
        IO.print s!"ORIENT: {eqn} (by priority)\n\n"
        return (EqValue (ValueVar p₂ n₂) (ValueVar p₁ n₁), true)
      else
        return (eqn, false)
  | EqType (TypeVar p₁ n₁) (TypeVar p₂ n₂) =>
      if p₂ > p₁ then do
        IO.print s!"ORIENT: {eqn} (by priority)\n\n"
        return (EqType (TypeVar p₂ n₂) (TypeVar p₁ n₁), true)
      else
        return (eqn, false)
  -- Orient to substitute variables with full terms
  | EqValue v₁ (ValueVar p₂ n₂) => do
      IO.print s!"ORIENT: {eqn}\n\n"
      return (EqValue (ValueVar p₂ n₂) v₁, true)
  | EqType t₁ (TypeVar p₂ n₂) => do
      IO.print s!"ORIENT: {eqn}\n\n"
      return (EqType (TypeVar p₂ n₂) t₁, true)
  | _ =>
      return (eqn, false)

private def orient (equations: List UEq): IO (List UEq × Bool) :=
  equations.foldlM
    (fun (done, b) eqn => do
      let (eqn, b') ← orientOne eqn
      return (done ++ [eqn], b || b'))
    ([], false)

-- ERASE: remove equation [x = x] where x is a variable

private def eraseFilter: UEq → Bool
  | EqValue (ValueVar p₁ n₁) (ValueVar p₂ n₂) =>
      n₁ = n₂
  | EqType (TypeVar p₁ n₁) (TypeVar p₂ n₂) =>
      n₁ = n₂
  | _ =>
      false

private def erase (equations: List UEq): List UEq × Bool :=
  let equations' := equations.filter (fun eq => ! eraseFilter eq)
  (equations', equations'.length != equations.length)

-- REDUCE: reduce [op = op'] to equality of arguments and return values (or no
-- solution if the constructors differ)

private def reduceOne (eqn: UEq): IO (Option (List UEq × Bool)) :=
  match eqn with
  | EqOp (OpKnown mnemonic₁ vals₁ rets₁) (OpKnown mnemonic₂ vals₂ rets₂) => do
      if mnemonic₁ = mnemonic₂
          ∧ vals₁.length = vals₂.length
          ∧ rets₁.length = rets₂.length then
        IO.print s!"REDUCE: {eqn}\n\n"
        let zipPairs := fun (v₁, t₁) (v₂, t₂) =>
          [EqValue v₁ v₂, EqType t₁ t₂]
        return some (List.join (
            List.map₂ zipPairs vals₁ vals₂ ++
            List.map₂ zipPairs rets₁ rets₂),
          true)
      else
        return none
  | eqn =>
      return some ([eqn], false)

private def reduce (equations: List UEq): IO (Option (List UEq × Bool)) :=
  equations.foldlM
    (fun acc eqn => do
       match ← reduceOne eqn with
       | some (equations, b') =>
          return acc.bind fun (done, b) => some (done ++ equations, b || b')
       | none =>
          return none)
    $ some ([], false)

-- ELIM: eliminate [x = t] by substituting x if it's used elsewhere and does
-- not occur in t (no solution if x occurs in t)

private def elimAt (equations: List UEq) (n: Nat):
    IO (Option (List UEq × List UEq)) := do

  if H: n < equations.length then
    let eqn := equations.get ⟨n, H⟩
    let others := (equations.enum.filter (·.1 ≠ n)).map (·.snd)

    if let EqValue (ValueVar p₁ n₁) v₂ := eqn then
      if v₂.occurs n₁ then do
        IO.println s!"Equation {eqn} has a cycle!"
        return none -- cycle
      else if others.any (·.occurs n₁) then do
        IO.print s!"SUBSTITUTE: {eqn}\n\n"
        return some (others.map (·.substValue n₁ v₂), [eqn])

    else if let EqType (TypeVar p₁ n₁) t₂ := eqn then
      if t₂.occurs n₁ then do
        IO.println s!"Equation {eqn} has a cycle!"
        return none -- cycle
      else if others.any (·.occurs n₁) then do
        IO.print s!"SUBSTITUTE: {eqn}\n\n"
        return some (others.map (·.substType n₁ t₂), [eqn])

  return some (equations, [])

private def elim (equations: List UEq): IO (Option (List UEq × List UEq)) :=
  (List.range equations.length).foldlM
    (fun acc n => do
      if let some (equations, substs) := acc then
        if let some (equations', substs') := ← elimAt equations n then
          return (equations', substs ++ substs')
      return none)
    $ some (equations, [])

-- Apply a single round of transformations
def Unification.simplify (u: Unification): IO (Option (Unification×Bool)) := do
  -- Orient all rules
  let (equations, b) ← orient u.equations
  if b then return some ({u with equations := equations}, true) else
  -- Substitute all intermediate variables
  match (← elim equations) with
  | some (equations, []) =>
      -- Match arguments and return values of common operations
      match (← reduce equations) with
      | some (equations, b) => return some ({u with equations := equations}, b)
      | none => return none
  | some (equations, substs) =>
      return some ({ u with equations := equations, substs := u.substs++substs},
                   true)
  | none =>
      return none

-- Solve the problem by normalizing for the transformations. We don't really
-- mind whether the function is total, but the bound avoids infinite loops in
-- case something goes wrong.
def Unification.solve (u: Unification) (steps: Nat := 100):
    IO (Option Unification) := do
  IO.print s!"{u}\n\n"
  match ← u.simplify with
  | some (u, b) =>
      -- Clean up after every step
      let u: Unification := {u with equations := (erase u.equations).fst}
      if b then
        if steps > 0 then
          let u ← u.solve (steps-1)
          return u
        else
          IO.println s!"Exhausted limited amount of steps."
          return u
      return u
  | none =>
      IO.println s!"Problem has no solution!"
      return none

def Unification.applyOnOp (solved_u: Unification) (op: MOp): MOp :=
  List.foldl
    (fun op eqn =>
      match eqn with
      | EqValue (ValueVar p₁ n₁) v₂ => op.substValue n₁ v₂
      | EqType (TypeVar p₁ n₁) t₂ => op.substType n₁ t₂
      | _ => op)
    op (solved_u.substs ++ solved_u.equations)

/-
### Basic example

Here, we consider an under-specified [x*2] pattern (ex_root) that we presumably
want to turn into [x+x]. The pattern doesn't specify that x is an i32 as this
is implicit, and we uncover this fact by unifying with the general shape of a
multiplication operation (mul_pattern), supposedly obtained from IRDL.
-/

-- $op_res:!T = "arith.mul"($op_x:!T, $op_y:!T)
private def mul_pattern: MOp :=
  OpKnown "arith.mul"
    [(ValueVar 2 "op_x", TypeVar 2 "T"), (ValueVar 2 "op_y", TypeVar 2 "T")]
    [(ValueVar 2 "op_res", TypeVar 2 "T")]

-- The following equality could be derived from PDL:
--   %two = pdl.value 2: i32
private def ex_two: UEq :=
  EqValue (ValueVar 0 "two") (ValueConst (MLIRType.int 32) 2)

-- The following matching pattern could be derived from PDL:
--   %x = pdl.value
--   %root = "arith.mul"(%x, %two)
-- (%x is implicit, while %x_T, %_0 and %_0_T are automatically generated)
private def ex_root: MOp :=
  OpKnown "arith.mul"
    [(ValueVar 0 "x", TypeVar 1 "x_T"),
     (ValueVar 0 "two", TypeConst (MLIRType.int 32))]
    [(ValueVar 1 "_res0", TypeVar 1 "_res0_T")]

private def mul_example: Unification :=
  { Unification.empty with equations := [ex_two, EqOp mul_pattern ex_root] }

#eval show IO Unit from do
  let u ← mul_example.solve
  let stmt := u.get!.applyOnOp ex_root
  IO.println s!"Theorem input:\n{stmt}"
