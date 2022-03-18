/-
## SSA environment

This file implements the SSA environment which maps variables names from
different scopes to explicitly-typed values. It is, conceptually, a map
`SSAVal → (α: MLIRTy) × α` for each scope.

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
    => Use default values! Data is always inhabited.
 2. Improve the typing: auto-simplify τ.eval and the resulting casts
 3. Set up infrastructure to unpack MLIRTy types into Lean native stuff

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

import MLIRSemantics.Fitree

import MLIR.AST
open MLIR.AST


-- Additional definitions for MLIR.AST types

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

deriving instance DecidableEq for Dimension

def MLIRTy.beq (t1 t2: MLIRTy): Bool :=
  match t1, t2 with
  | MLIRTy.fn a1 b1, MLIRTy.fn a2 b2 =>
      beq a1 a2 && beq b1 b2
  | MLIRTy.int n1, MLIRTy.int n2 =>
      n1 == n2
  | MLIRTy.float n1, MLIRTy.float n2 =>
      n1 == n2
  | MLIRTy.tuple [], MLIRTy.tuple [] =>
      true
  | MLIRTy.tuple (t1::l1), MLIRTy.tuple (t2::l2) =>
      beq t1 t2 && beq (MLIRTy.tuple l1) (MLIRTy.tuple l2)
  | MLIRTy.vector l1 t1, MLIRTy.vector l2 t2 =>
      l1 == l2 && beq t1 t2
  | MLIRTy.tensor l1 t1, MLIRTy.tensor l2 t2 =>
      l1 == l2 && beq t1 t2
  | MLIRTy.user n1, MLIRTy.user n2 =>
      n1 == n2
  | _, _ =>
      false

def MLIRTy.decEq (t1 t2: MLIRTy): Decidable (Eq t1 t2) :=
  if MLIRTy.beq t1 t2 then isTrue sorry else isFalse sorry

instance: DecidableEq MLIRTy :=
  MLIRTy.decEq


-- Evaluation of MLIR types
-- TODO: Not all MLIRTy types are correctly evaluated

@[simp, inline]
def MLIR.AST.MLIRTy.eval: MLIRTy → Type
  | MLIRTy.fn τ₁ τ₂ => τ₁.eval → τ₂.eval
  | MLIRTy.int _ => Int
  | MLIRTy.float _ => Float
  | MLIRTy.tuple [] => Unit
  | MLIRTy.tuple [τ] => τ.eval
  | MLIRTy.tuple (τ::l) => τ.eval × (MLIRTy.tuple l).eval
  | MLIRTy.vector _ _ => Unit /- todo -/
  | MLIRTy.tensor _ _ => Unit /- todo -/
  | MLIRTy.user _ => Unit /- todo -/


-- SSAScope

def SSAScope :=
  List (SSAVal × (τ: MLIRTy) × τ.eval)

@[simp]
def SSAScope.get (name: SSAVal): SSAScope → (τ: MLIRTy) → Option τ.eval
  | [], _ => none
  | ⟨name', τ', v'⟩ :: l, τ =>
      if H: name' = name && τ' = τ then
        some (cast (by simp at H; simp [H]) v')
      else get name l τ

@[simp]
def SSAScope.set (name: SSAVal) (τ: MLIRTy) (v: τ.eval):
      SSAScope → SSAScope
  | [] => [⟨name, τ, v⟩]
  | ⟨name', τ', v'⟩ :: l =>
      if name' = name
      then ⟨name', τ, v⟩ :: l
      else ⟨name', τ', v'⟩ :: set name τ v l

/- Maybe useful in the future, for proofs
def SSAScope.has (name: SSAVal) (l: SSAScope): Bool :=
  l.any (fun ⟨name', _, _⟩ => name' == name)

def SSAScope.free (name: SSAVal) (l: SSAScope): Bool :=
  l.all (fun ⟨name', _, _⟩ => name' != name)

def SSAScope.maps (l: SSAScope) (name: SSAVal) (τ: MLIRTy) (v: τ.eval) :=
  l.Mem ⟨name, τ, v⟩ -/


-- SSAEnv

def SSAEnv :=
  List SSAScope

@[simp]
def SSAEnv.get (name: SSAVal) (τ: MLIRTy): SSAEnv → Option τ.eval
  | [] => none
  | l :: s =>
      match l.get name τ with
      | none => get name τ s
      | some v => v

@[simp]
def SSAEnv.set (name: SSAVal) (τ: MLIRTy) (v: τ.eval): SSAEnv → SSAEnv
  | [] => [] -- cannot happen in practice
  | l :: s => l.set name τ v :: s

/- Useful for proofs
def SSAEnv.refines (new old: SSAEnv) :=
  ∀ ⦃name τ v⦄, old.get name τ = some v → new.get name τ = some v

instance: LT SSAEnv where
  lt := SSAEnv.refines -/


-- Interactions manipulating the environment

inductive SSAEnvE: Type → Type _ where
  | Get: (τ: MLIRTy) → [Inhabited τ.eval] → SSAVal → SSAEnvE τ.eval
  | Set: (τ: MLIRTy) → SSAVal → τ.eval → SSAEnvE Unit

def SSAEnvE.handle: SSAEnvE ~> StateT SSAEnv (Fitree PVoid) :=
  λ _ e env =>
    match e with
    | Get τ name =>
        match env.get name τ with
        | some v => return (v, env)
        | none => return (default, env)
    | Set τ name v =>
        return ((), env.set name τ v)


-- Tests

private def example_env (l: SSAScope) (s: SSAEnv): SSAEnv :=
  (⟨"%0", MLIRTy.int 32, (42:Int)⟩ ::
   ⟨"%1", MLIRTy.int 32, (7:Int)⟩ ::
   l) ::
  (⟨"%3", MLIRTy.int 32, (12:Int)⟩ ::
   ⟨"%r", MLIRTy.float 32, (-1.7e3:Float)⟩ ::
   []) ::
  s

example: ∀ l s, SSAEnv.get "%1" (MLIRTy.int 32) (example_env l s)
    = some (7:Int) :=
  by simp [cast_eq]
