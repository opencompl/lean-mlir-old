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
-/

import MLIR.Semantics.Fitree
import MLIR.Semantics.Types

import MLIR.AST
open MLIR.AST


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

instance: LE SSAEnv where
  le := SSAEnv.refines -/


-- Interactions manipulating the environment

inductive SSAEnvE: Type → Type _ where
  | Get: (τ: MLIRTy) → [Inhabited τ.eval] → SSAVal → SSAEnvE τ.eval
  | Set: (τ: MLIRTy) → SSAVal → τ.eval → SSAEnvE Unit

@[simp_itree]
def SSAEnvE.handle {E}: SSAEnvE ~> StateT SSAEnv (Fitree E) :=
  fun _ e env =>
    match e with
    | Get τ name =>
        match env.get name τ with
        | some v => return (v, env)
        | none => return (default, env)
    | Set τ name v =>
        return ((), env.set name τ v)

@[simp_itree]
def SSAEnv.set? {E} [Member SSAEnvE E]
    (τ: MLIRTy) (name?: Option SSAVal) (v: τ.eval): Fitree E Unit :=
  match name? with
  | some name =>
      Fitree.trigger (SSAEnvE.Set τ name v)
  | none =>
      return ()

-- In-context handler interpreting (SSAEnvE +' E ~> E)

@[simp_itree]
private def stateT_defaultHandler E: E ~> StateT SSAEnv (Fitree E) :=
  fun _ e m => do
    let r <- Fitree.trigger e;
    return (r, m)

def interp_ssa {E R} (t: Fitree (SSAEnvE +' E) R):
    StateT SSAEnv (Fitree E) R :=
  interp_state (case_ SSAEnvE.handle (stateT_defaultHandler E)) t
