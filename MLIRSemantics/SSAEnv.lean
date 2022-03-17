/-
## SSA environment

This file implements the SSA environment which maps variables names from
different scopes to explicitly-typed values. It is, conceptually, a map
`SSAVal → (α: Type) × α` for each scope.

The main concern with such state is the definition, maintainance and use of the
property that the values are defined only once, allowing us to study sections
of the code while skipping/omitting operations very freely. Bundling uniqueness
invariants with the data structure would result in very tedious and context-
dependent maintenance, which is not a friendly option.

Instead, we ignore the SSA constraints when defining semantics and interpreting
programs, assuming the language allows shadowing and overriding values (of
course, valid programs won't do that). We only define SSA constraints later on
to prove that transformations are context-independent.

`SSAScope` implements a single scope as a list of name/value pairs and supports
edition.

`SSAEnv` implements a stack of scopes. New scopes are created when entering
regions. Here again nothing prevents a region from shadowing variables or
accessing out-of-scope values (in the case of an isolated region), but we only
care when proving the correction of transformations.

TODO:
 1. How to guarantee that read values exist and have the proper type without
    going too deep into proofs?
 2. Document events

Linking definitions and operations:
 1. run (%a = 2) in environment env :: whatever
 2. this returns () and env[%a=2] :: whatever
 3. run (return %a) in environment env[%a=2] :: whatever
 4. we can thus prove this returns 2
 5. run (%b = %ext) in environment env[%a=2] :: whatever
 6. result depends on whether %ext was defined in env, which we can't just
    assume since we want theorems that work *in context*

Value scoping: operation o1 dominates operation o2 if:
 - o2 is in the same block as o1 after o1, or
 - o2 is in a non-OpTrait::IsolatedFromAbove region dominated by o1
 - we don't handle concurrent semantics :x
Block scoping: operation o1 can branch to block b2 if:
 - b2 is in the same region as the block owning o1
-/

import MLIR.AST
open MLIR.AST

-- Shortcuts for SSAVal

instance: Coe String SSAVal where
  coe := SSAVal.SSAVal

def SSAVal.decEq (v1 v2: SSAVal): Decidable (Eq v1 v2) :=
  match v1, v2 with
  | SSAVal.SSAVal s1, SSAVal.SSAVal s2 =>
      dite (s1 = s2)
        (fun h => isTrue (by simp [h]))
        (fun h => isFalse (by simp [h]))

instance: DecidableEq SSAVal :=
  SSAVal.decEq

-- SSAScope

def SSAScope :=
  List (SSAVal × (α: Type) × α)

@[simp]
def SSAScope.get (name: SSAVal): SSAScope → Option ((α: Type) × α)
  | [] => none
  | ⟨name', α, v⟩ :: l => if name' = name then some ⟨α, v⟩ else get name l

@[simp]
def SSAScope.set {α} (name: SSAVal) (v: α): SSAScope → SSAScope
  | [] => [⟨name, α, v⟩]
  | ⟨name', β, v'⟩ :: l =>
      if name' = name
      then ⟨name', α, v⟩ :: l
      else ⟨name', β, v'⟩ :: set name v l

/- Maybe useful in the future, for proofs
def SSAScope.has (name: SSAVal) (l: SSAScope): Bool :=
  l.any (fun ⟨name', _, _⟩ => name' == name)

def SSAScope.free (name: SSAVal) (l: SSAScope): Bool :=
  l.all (fun ⟨name', _, _⟩ => name' != name)

def SSAScope.maps (l: SSAScope) {α} (name: SSAVal) (value: α): Prop :=
  l.Mem ⟨name, α, value⟩ -/

-- SSAEnv

def SSAEnv :=
  List SSAScope

@[simp]
def SSAEnv.get (name: SSAVal): SSAEnv → Option ((α: Type) × α)
  | [] => none
  | l :: s =>
      match l.get name with
      | none => get name s
      | some v => v

@[simp]
def SSAEnv.set {α} (name: SSAVal) (v: α): SSAEnv → SSAEnv
  | [] => [] -- cannot happen in practice
  | l :: s => l.set name v :: s

/- Useful for proofs
def SSAEnv.refines (new old: SSAEnv) :=
  ∀ ⦃name v⦄, old.get name = some v → new.get name = some v

instance: LT SSAEnv where
  lt := SSAEnv.refines -/

-- Interactions manipulating the environment

inductive SSAEnvE: Type → Type where
  | Get: (α: Type) → SSAVal → SSAEnvE α


-- Tests

private def example_env (l: SSAScope) (s: SSAEnv): SSAEnv :=
  (⟨"%0", UInt32, 42⟩ :: ⟨"%1", UInt32, 7⟩ :: l) ::
  (⟨"%3", UInt32, 12⟩ :: ⟨"%res", String, "abc"⟩ :: []) ::
  s

example: ∀ l s, SSAEnv.get "%1" (example_env l s) = some ⟨UInt32, 7⟩ :=
  by simp


