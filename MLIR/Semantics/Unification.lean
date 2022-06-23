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
problem. They can be constructed mistyped, but we check them.
-/

def UEq := MTerm × MTerm

-- Common instances

deriving instance Inhabited for UEq

def UEq.str: UEq → String
  | (left, right) => s!"{left} ≡ {right}"

instance: ToString UEq where
  toString := UEq.str

def UEq.inferSort (eq: UEq): Option MSort := do
  let s₁ ← eq.1.inferSort
  let s₂ ← eq.2.inferSort
  if s₁ = s₂ then some s₁ else none

-- Extensions of functions on matching patterns

def UEq.vars (eq: UEq): List String :=
  eq.1.vars ++ eq.2.vars

def UEq.occurs (eq: UEq) (name: String): Bool :=
  eq.1.occurs name || eq.2.occurs name

def UEq.subst (eq: UEq) (name: String) (repl: MTerm): UEq :=
  (eq.1.subst name repl, eq.2.subst name repl)

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
  -- TODO: More specific definition of Unification.substs?
  substs: List UEq := []

-- Common instances

deriving instance Inhabited for Unification

def Unification.empty: Unification :=
  default

def Unification.str (u: Unification): String :=
  if u.equations.isEmpty then
    "(empty unification problem: no equations)"
  else
    "\n".intercalate (u.equations.map toString)

instance: ToString Unification where
  toString := Unification.str

def Unification.repr (u: Unification): String :=
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
  4. If we have `constructor(a1, ... ak) = constructor(b1, ... bk), we add
     equalities `ai = bi`.
  5. Make sure that we have substitution priority right.
-/

-- ORIENT: turn [t = x] (where t not a variable) into [x = t]
-- We also orient equations [y = x] if y is a variable with a higher
-- substitution priority than x.

private def orientOne (eq: UEq): IO (UEq × Bool) :=
  match eq with
  -- Orient to substitute automatically generated names first
  | (.Var p₁ n₁ s₁, .Var p₂ n₂ s₂) =>
      if p₂ > p₁ then do
        IO.print s!"ORIENT: {eq} (by priority)\n\n"
        return ((.Var p₂ n₂ s₂, .Var p₁ n₁ s₁), true)
      else
        return (eq, false)
  -- Orient to substitute variables with full terms
  | (t₁, .Var p₂ n₂ s₂) => do
      IO.print s!"ORIENT: {eq}\n\n"
      return ((.Var p₂ n₂ s₂, t₁), true)
  | _ =>
      return (eq, false)

private def orient (equations: List UEq): IO (List UEq × Bool) :=
  equations.foldlM
    (fun (done, b) eq => do
      let (eq, b') ← orientOne eq
      return (done ++ [eq], b || b'))
    ([], false)

-- ERASE: remove equation [x = x] where x is a variable

private def eraseFilter: UEq → Bool
  | (.Var p₁ n₁ s₁, .Var p₂ n₂ s₂) =>
      -- Keep mistyped equations so they cause errors when analyzed later
      n₁ = n₂ && s₁ = s₂
  | _ =>
      false

private def erase (equations: List UEq): List UEq × Bool :=
  let equations' := equations.filter (! eraseFilter ·)
  (equations', equations'.length != equations.length)

-- REDUCE: reduce [ctor args... = ctor' args'...] to equality of arguments (or
-- no solution if the constructors or argument counts differ)

private def reduceOne (eq: UEq): IO (Option (List UEq × Bool)) :=
  match eq with
  | (.App ctor₁ args₁, .App ctor₂ args₂) => do
      if MCtor.eq ctor₁ ctor₂ && args₁.length == args₂.length then
        IO.print s!"REDUCE: {eq}\n\n"
        return some (List.zip args₁ args₂, true)
      else
        return none
  | eq =>
      return some ([eq], false)

private def reduce (equations: List UEq): IO (Option (List UEq × Bool)) :=
  equations.foldlM
    (fun acc eq => do
       match ← reduceOne eq with
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
    let eq := equations.get ⟨n, H⟩
    let others := (equations.enum.filter (·.1 ≠ n)).map (·.snd)

    if let (.Var p₁ n₁ s₁, t₂) := eq then
      if t₂.occurs n₁ then do
        IO.println s!"Equation {eq} has a cycle!"
        return none -- cycle
      else if others.any (·.occurs n₁) then do
        IO.print s!"SUBSTITUTE: {eq}\n\n"
        return some (others.map (·.subst n₁ t₂), [eq])

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

def Unification.applyOnTerm (solved_u: Unification) (t: MTerm): MTerm :=
  List.foldl
    (fun t eq =>
      match eq with
      | (.Var p₁ n₁ s₁, t₂) => t.subst n₁ t₂
      | _ => t)
    t (solved_u.substs ++ solved_u.equations)

/-
### Basic example

Here, we consider an under-specified [x*2] pattern (ex_root) that we presumably
want to turn into [x+x]. The pattern doesn't specify that x is an i32 as this
is implicit, and we uncover this fact by unifying with the general shape of a
multiplication operation (mul_pattern), supposedly obtained from IRDL.
-/

-- %op_res:!T = "arith.mul"(%op_x:!T, %op_y:!T)
private def mul_pattern: MTerm :=
  .App .OP [
    .ConstString "arith.mul",
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "op_x" .MSSAVal, .Var 2 "T" .MMLIRType],
      .App .OPERAND [.Var 2 "op_y" .MSSAVal, .Var 2 "T" .MMLIRType]],
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "op_res" .MSSAVal, .Var 2 "T" .MMLIRType]]
  ]

-- TODO: We don't have attributes, so we assume there is an "arith.two"
-- operation that always returns 2
-- %two:i32 = "arith.two"()
private def ex_two: MTerm :=
  .App .OP [
    .ConstString "arith.two",
    .App (.LIST .MOperand) [],
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "two" .MSSAVal, .ConstMLIRType .i32]]
  ]

-- The following matching pattern could be derived from PDL:
--   %x = pdl.value
--   %root = "arith.mul"(%x, %two)
-- (%x is implicit, while %x_T, %_0 and %_0_T are automatically generated)
private def ex_root: MTerm :=
  .App .OP [
    .ConstString "arith.mul",
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 0 "x" .MSSAVal, .Var 1 "x_T" .MMLIRType],
      .App .OPERAND [.Var 0 "two" .MSSAVal, .ConstMLIRType .i32]],
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 1 "_res0" .MSSAVal, .Var 1 "_res0_T" .MMLIRType]]
  ]

-- TODO: How to mix ex_two in there?
private def mul_example: Unification :=
  { Unification.empty with equations := [(mul_pattern, ex_root)] }

#eval show IO Unit from do
  let u ← mul_example.solve
  let stmt := u.get!.applyOnTerm ex_root
  IO.println s!"Theorem input:\n{stmt}"
