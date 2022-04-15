/-
## Rewriting MLIR fragments with PDL

This file implements some framework support for PDL rewrites, which is used to
formalize and prove rewrite operations.

This formalization works by treating a PDL rewrite as a first-order unification
problem. In order to prove a rewrite, we first consider a most general solution
to the unification problem, and then add any other constraints as hypotheses of
the correction theorem.

The general idea is to take as input a PDL operation, and output a template
theorem that looks like this:

```lean4
-- (unification variables)
∀ (value_1: Value) (type_1: Type) (op_1: Operation),
  -- (non-unification-based constraints)
  is_float_type type_1 →

  semantics_of [mlir_bb|
    -- (symbolic pattern, almost verbatim from the `pdl.pattern` description)
    op_1 (value_1: type_1) 2
  ] =
  semantics_of [mlir_bb|
    -- (rewritten version of above)
    output_op_1 ...
  ]
```

For readability/accessibility reasons, this statement would probably be
synthetized into Lean code rather than being a general theorem instantiated
with that particular PDL rewrite. The second option would require considering
"any matching basic block" and then painstakingly proving that such a match has
the structure underlined by the PDL rewrite, which seems annoying.

Operations of PDL that we should partially or totally express:

* `pdl.attribute`:
  Used as unification variable. TODO: should support type/value requirements
* `pdl.erase`:
  One of the rewriting actions
* `pdl.operand`:
  Used as unification variable. TODO: should support type requirement
* `pdl.operation`:
  Used as unification variable and as rewriting output.
  TODO: should support type/value/attributes, for fairly obvious reasons.
  Ranges are not important for now.
* `pdl.pattern`:
  Input operation for the whole rewriting workflow.
* `pdl.replace`:
  One of the rewriting actions. TODO: should support both values and operations
* `pdl.result`:
  Utility to reference results from `pdl.operation` pointers
* `pdl.rewrite`:
  One of the main components of the specification. We might not care about the
  root selection since we don't implement the pattern matching/search.
* `pdl.type`:
  Used as unification variable.

Operations of PDL that we don't try to express (yet):

* `pdl.operands`, `pdl.results`, `pdl.types`
  Anything related to ranges
* `pdl.apply_native_constraint`, `pdl.apply_native_rewrite`:
  This is possible in principle, but requires the native constraints/rewrites
  to be rewritten in Lean, which is slightly inconvenient.
* Use of native rewrite function in `pdl.rewrite`:
  Same as above
-/

import MLIRSemantics.Types
import MLIR.AST
open MLIR.AST


/-
### Syntax of the unification problem

For this problem, we have different sorts: type, attributes, values (operands)
and operations. TODO
-/

inductive UType :=
  -- A type variable
  | Var: String → UType
  -- A constant type
  | Const: MLIRTy → UType

inductive UAttr :=
  -- TODO: Unification with attributes

inductive UValue :=
  -- A value variable
  | Var: SSAVal → UValue
  -- A constant value
  | Const: (τ: MLIRTy) → τ.eval → UValue

inductive UOp :=
  -- An operation with a known mnemonic, unifiable arguments, and named return
  -- values (TODO: Types on retun values of UOp ?)
  | Known: String → List (UValue × UType) → List (UValue × UType) → UOp

inductive UEq :=
  | EqType: UType → UType → UEq
  | EqValue: UValue → UValue → UEq
  | EqOp: UOp → UOp → UEq

structure Unification where mk::
  -- Current set of equations
  eqns: List UEq
  -- Substitutions performed so far
  substs: List UEq
deriving Inhabited

/- Occurrence check and substitution -/

def UType.occursType: UType → String → Bool
  | Var s, s' => s = s'
  | _, _ => false

def UType.substType (t: UType) (n₁: String) (t₂: UType): UType :=
  match t with
  | Var s =>
      if s = n₁ then t₂ else Var s
  | t => t

def UValue.occursValue: UValue → SSAVal → Bool
  | Var s, s' => s = s'
  | _, _ => false

def UValue.substValue (v: UValue) (n₁: SSAVal) (v₂: UValue): UValue :=
  match v with
  | Var s =>
       if s = n₁ then v₂ else Var s
  | v => v

def UOp.occursValue: UOp → SSAVal → Bool
  | Known _ vals _, x => vals.any (·.fst.occursValue x)

def UOp.substValue (op: UOp) (n₁: SSAVal) (v₂: UValue): UOp :=
  match op with
  | Known name vals rets =>
      Known name (vals.map fun (v,t) => (v.substValue n₁ v₂, t))
                 (rets.map fun (v,t) => (v.substValue n₁ v₂, t))

def UOp.occursType: UOp → String → Bool
  | Known _ vals _, t => vals.any (·.snd.occursType t)

def UOp.substType (op: UOp) (n₁: String) (t₂: UType): UOp :=
  match op with
  | Known name vals rets =>
      Known name (vals.map fun (v,t) => (v, t.substType n₁ t₂))
                 (rets.map fun (v,t) => (v, t.substType n₁ t₂))

def UEq.occursValue: UEq → SSAVal → Bool
  | EqType t₁ t₂, s => false
  | EqValue v₁ v₂, s => v₁.occursValue s ∨ v₂.occursValue s
  | EqOp op₁ op₂, s => op₁.occursValue s ∨ op₂.occursValue s

def UEq.substValue (n₁: SSAVal) (v₂: UValue): UEq → UEq
  | EqType t₁ t₂ => EqType t₁ t₂
  | EqValue v₁ _v₂ => EqValue (v₁.substValue n₁ v₂) (_v₂.substValue n₁ v₂)
  | EqOp op₁ op₂ => EqOp (op₁.substValue n₁ v₂) (op₂.substValue n₁ v₂)

def UEq.occursType: UEq → String → Bool
  | EqType t₁ t₂, s => t₁.occursType s ∨ t₂.occursType s
  | EqValue v₁ v₂, s => false
  | EqOp op₁ op₂, s => op₁.occursType s ∨ op₂.occursType s

def UEq.substType (n₁: String) (t₂: UType): UEq → UEq
  | EqType t₁ _t₂ => EqType (t₁.substType n₁ t₂) (_t₂.substType n₁ t₂)
  | EqValue v₁ v₂ => EqValue v₁ v₂
  | EqOp op₁ op₂ => EqOp (op₁.substType n₁ t₂) (op₂.substType n₁ t₂)


/- String representations -/

def UType.str: UType → String
  | Var var => "!" ++ var
  | Const τ => toString τ

instance: ToString UType := ⟨UType.str⟩

-- TODO: String representation for any τ.eval
def UValue.str: UValue → String
  | Var var => toString var
  | Const type const => s!"TODO:{type}"

instance: ToString UValue := ⟨UValue.str⟩

def UOp.str: UOp → String
  | Known name vals rets =>
      let rets := ", ".intercalate (rets.map fun (v,t) => s!"{v}:{t}")
      let vals := ", ".intercalate (vals.map fun (v,t) => s!"{v}:{t}")
      s!"{rets} = \"{name}\"({vals})"

instance: ToString UOp := ⟨UOp.str⟩

def UEq.str: UEq → String
  | EqType t₁ t₂   => s!"{t₁} = {t₂}"
  | EqValue v₁ v₂  => s!"{v₁} = {v₂}"
  | EqOp o₁ o₂     => s!"[{o₁}]\n  = [{o₂}]"

instance: ToString UEq := ⟨UEq.str⟩

def Unification.str: Unification → String := fun u =>
  "\n".intercalate (u.eqns.map toString)

instance: ToString Unification := ⟨Unification.str⟩


/-
### Unification algorithm

We use a naive unification algorithm. Given the size of the rewrites at play,
we don't really care about performance, and instead prefer a more natural
traveral of the structure that leads to more intuitive logs and error messages.

This is algorithm 1 of [1], which essentially normalizes the set of equations
through a number of unifier-preserving transformations.

[1] Martelli, Alberto, and Ugo Montanari. "An efficient unification algorithm."
    ACM Transactions on Programming Languages and Systems (TOPLAS) 4.2 (1982):
    258-282.
-/

open UValue UType UOp UEq

-- Transformation (a): turn [t = x] (t not a variable) into [x = t]

private def orient_one (eqn: UEq): IO (UEq × Bool) :=
  match eqn with
  | EqType (UType.Var v₁) (UType.Var v₂) =>
      return (eqn, false)
  | EqType t₁ (UType.Var v₂) => do
      IO.print s!"Orient: {eqn}\n\n"
      return (EqType (Var v₂) t₁, true)
  | EqValue (UValue.Var v₁) (UValue.Var v₂) =>
      return (eqn, false)
  | EqValue t₂ (UValue.Var v₂) => do
      IO.print s!"Orient: {eqn}\n\n"
      return (EqValue (Var v₂) t₂, true)
  | _ =>
      return (eqn, false)

private def orient (eqns: List UEq): IO (List UEq × Bool) :=
  eqns.foldlM
    (fun (trs, b) eqn => do
      let (eqn, b') ← orient_one eqn
      return (trs ++ [eqn], b || b'))
    ([], false)

-- Transformation (b): erase [x = x] (x a variable)

private def erase_filter: UEq → Bool
  | EqType (UType.Var v₁) (UType.Var v₂) =>
      v₁ = v₂
  | EqValue (UValue.Var v₁) (UValue.Var v₂) =>
      v₁ = v₂
  | _ =>
      false

private def erase (eqns: List UEq): List UEq × Bool :=
  let eqns' := eqns.filter (fun eq => ! erase_filter eq)
  (eqns', eqns'.length != eqns.length)

-- Transformation (c): reduce [t = t'], where t and t' are constructed, to
-- equality of arguments (or no solution if the constructors differ)
-- TODO: Also break up interesting type and value equalities

private def reduce_one (eqn: UEq): IO (Option (List UEq × Bool)) :=
  match eqn with
  | EqOp (Known name₁ vals₁ rets₁) (Known name₂ vals₂ rets₂) => do
      if name₁ = name₂
       ∧ vals₁.length = vals₂.length
       ∧ rets₁.length = rets₂.length then
        IO.print s!"Reduce: {eqn}\n\n"
        return some (
          List.map₂ (fun (v₁,t₁) (v₂,t₂) => EqValue v₁ v₂) vals₁ vals₂ ++
          List.map₂ (fun (v₁,t₁) (v₂,t₂) => EqType t₁ t₂) vals₁ vals₂ ++
          List.map₂ (fun (v₁,t₁) (v₂,t₂) => EqValue v₁ v₂) rets₁ rets₂ ++
          List.map₂ (fun (v₁,t₁) (v₂,t₂) => EqType t₁ t₂) rets₁ rets₂, true)
      else
        return none
  | eq =>
      return some ([eq], false)

private def reduce (eqns: List UEq): IO (Option (List UEq × Bool)) :=
  eqns.foldlM
    (fun acc eqn => do
       match ← reduce_one eqn with
       | some (eqns, b') =>
          return acc.bind fun (trs, b) => some (trs ++ eqns, b || b')
       | none =>
          return none)
    $ some ([], false)

-- Transformation (d): eliminate [x = t] by substituting x if it's used
-- elsewhere and does not occur in t

private def elim_at (eqns: List UEq) (n: Nat):
    IO (Option (List UEq × List UEq)) :=
  if H: n < eqns.length then
    let eqn := eqns.get ⟨n, H⟩
    let others := (eqns.enum.filter (·.1 ≠ n)).map (·.snd)

    match eqn with
    | EqType (UType.Var v₁) t₂ =>
        if t₂.occursType v₁ then do
          IO.println s!"Equation {eqn} has a cycle!"
          return none -- cycle
        else if eqns.enum.any (fun (j,eqn) => j ≠ n ∧ eqn.occursType v₁) then do
          IO.print s!"Substitute: {eqn}\n\n"
          return some (others.map (·.substType v₁ t₂), [eqn])
        else
          return some (eqns, [])
    | EqValue (UValue.Var v₁) t₂ =>
        if t₂.occursValue v₁ then do
          IO.println s!"Equation {eqn} has a cycle!"
          return none -- cycle
        else if eqns.enum.any (fun (j,eqn) => j ≠ n ∧ eqn.occursValue v₁) then do
          IO.print s!"Substitute: {eqn}\n\n"
          return some (others.map (·.substValue v₁ t₂), [eqn])
        else
          return some (eqns, [])
    | _ =>
        return some (eqns, [])
  else
    return some (eqns, [])

private def elim (eqns: List UEq): IO (Option (List UEq × List UEq)) :=
  (List.range eqns.length).foldlM
    (fun acc n => do
      match acc with
      | some (eqns, substs) =>
          match ← elim_at eqns n with
          | some (eqns', substs') =>
              return some (eqns', substs ++ substs')
          | none =>
              return none
      | none =>
          return none)
    $ some (eqns, [])

-- Unification main loop: greedily applies each transformation in order

def Unification.simplify: Unification → IO (Option (Unification × Bool)) :=
  fun u => do
    -- Orient all rules
    let (eqns, b) ← orient u.eqns
    if b then return some ({u with eqns := eqns}, true) else
    -- Substitute all intermediate variables
    match (← elim eqns) with
    | some (eqns, []) =>
        -- Match arguments and return values of common operations
        match (← reduce eqns) with
        | some (eqns, b) => return some ({u with eqns := eqns}, b)
        | none => return none
    | some (eqns, substs) =>
        return some ({ eqns := eqns, substs := u.substs ++ substs}, true)
    | none =>
        return none

partial def Unification.solve (u: Unification) (n: Nat):
    IO (Option Unification) := do
  IO.print s!"{u}\n\n"
  match ← u.simplify with
  | some (u, b) =>
      -- Clean up after every step
      let u: Unification := {u with eqns := (erase u.eqns).fst}
      if b then
        if n > 0 then
          let u ← u.solve (n-1)
          return u
        else
          IO.println s!"Max iterations reached"
          return u
      return u
  | none =>
      IO.println s!"Problem has no solution!"
      return none

def Unification.apply (solved_u: Unification): UOp → UOp := fun op =>
  List.foldl (fun op eqn =>
      match eqn with
      | EqType (UType.Var n₁) t₂ => op.substType n₁ t₂
      | EqValue (UValue.Var n₁) v₂ => op.substValue n₁ v₂
      | _ => op)
    op (solved_u.substs ++ solved_u.eqns)


-- An example with a multiplication operation

private def mul_pattern: UOp :=
  ⟨"arith.mul", [(Var "op_x", Var "T"),
                 (Var "op_y", Var "T")],
                [(Var "op_res", Var "T")]⟩

-- %two = pdl.value 2: i32
private def ex_two: UEq :=
  EqValue (Var "two") (Const (MLIRTy.int 32) 2)

-- %x = pdl.value
-- %root = "arith.mul"(%x, %two)
-- (%x is implicit, while %x_T, %_0 and %_0_T are automatically inserted)
private def ex_root: UOp :=
  ⟨"arith.mul", [(Var "x", Var "x_T"),
                 (Var "two", Const (MLIRTy.int 32))],
                [(Var "_0", Var "_0_T")]⟩

private def mul_example: Unification :=
  { eqns := [ex_two, EqOp mul_pattern ex_root],
    substs := [] }

#eval show IO Unit from do
  let u ← mul_example.solve 999
  let stmt := u.get!.apply ex_root
  IO.println s!"Theorem input:\n{stmt}"

/- def substOne (identity: UEq): UEq → UEq :=
  match identity with
  | EqType (UType.Var n₁) t₂ => (·.substType n₁ t₂)
  | EqValue (UValue.Var n₁) v₂ => (·.substValue n₁ v₂)
  | _ => id

def substAll (identity: UEq): List UEq → List UEq :=
  List.map (substOne identity)

#eval show IO Unit from do
  let s ← mul_example.solve 10
  match s with
  | some s =>
      let u := List.foldl (fun u eqn => Unification.mk (substAll eqn u.eqns) []) mul_example s.eqns
      IO.println s!"Theorem input:\n{u}"
  | none =>
      return () -/
