/-
## SSA environment

This file implements the SSA environment which maps variables names from
different scopes to explicitly-typed values. It is, conceptually, a map
`SSAVal → (α: MLIRType δ) × α` for each scope.

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
import MLIR.Util.WriterT

import MLIR.AST
open MLIR.AST

section
variable {α σ: Type} {ε: σ → Type}

-- SSAScope

def SSAScope (δ: Dialect α σ ε) :=
  List (SSAVal × (τ: MLIRType δ) × τ.eval)

@[simp]
def SSAScope.get {δ: Dialect α σ ε} (name: SSAVal):
  SSAScope δ → (τ: MLIRType δ) → Option τ.eval
  | [], _ => none
  | ⟨name', τ', v'⟩ :: l, τ =>
      if H: name' = name && τ' = τ then
        some (cast (by simp at H; simp [H]) v')
      else get name l τ

@[simp]
def SSAScope.set {δ: Dialect α σ ε} (name: SSAVal) (τ: MLIRType δ) (v: τ.eval):
  SSAScope δ → SSAScope δ
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

def SSAEnv (δ: Dialect α σ ε) :=
  List (SSAScope δ)

-- An SSA environment with a single empty SSAScope
def SSAEnv.empty {δ: Dialect α σ ε}: SSAEnv δ := [[]]

@[simp]
def SSAEnv.get {δ: Dialect α σ ε} (name: SSAVal) (τ: MLIRType δ):
  SSAEnv δ → Option τ.eval
  | [] => none
  | l :: s =>
      match l.get name τ with
      | none => get name τ s
      | some v => v

@[simp]
def SSAEnv.set {δ: Dialect α σ ε} (name: SSAVal) (τ: MLIRType δ) (v: τ.eval):
  SSAEnv δ → SSAEnv δ
  | [] => [] -- cannot happen in practice
  | l :: s => l.set name τ v :: s

/- Useful for proofs
def SSAEnv.refines (new old: SSAEnv) :=
  ∀ ⦃name τ v⦄, old.get name τ = some v → new.get name τ = some v

instance: LE SSAEnv where
  le := SSAEnv.refines -/


-- Interactions manipulating the environment

inductive SSAEnvE (δ: Dialect α σ ε): Type → Type where
  | Get: (τ: MLIRType δ) → [Inhabited τ.eval] → SSAVal → SSAEnvE δ τ.eval
  | Set: (τ: MLIRType δ) → SSAVal → τ.eval → SSAEnvE δ Unit

@[simp_itree]
def SSAEnvE.handle {E}: SSAEnvE δ ~> StateT (SSAEnv δ) (Fitree E) :=
  fun _ e env =>
    match e with
    | Get τ name =>
        match env.get name τ with
        | some v => return (v, env)
        | none => return (default, env)
    | Set τ name v =>
        return (.unit, env.set name τ v)

def SSAEnvE.handleLogged {E}: SSAEnvE δ ~> WriterT (StateT (SSAEnv δ) (Fitree E)) :=
  fun _ e => do
    let env <- WriterT.lift StateT.get
    match e with
    | Get τ name =>
        match env.get name τ with
        | some v => do
            logWriterT s!"get {name} (={v}); "
            return v
        | none =>
            logWriterT s!"get {name} (not found!); "
            return default
    | Set τ name v =>
        logWriterT s!"set {name}={v}; "
        WriterT.lift $ StateT.set (env.set name τ v)
        return ()

@[simp_itree]
def SSAEnv.set? {E} {δ: Dialect α σ ε} [Member (SSAEnvE δ) E]
    (τ: MLIRType δ) (name?: Option SSAVal) (v: τ.eval): Fitree E Unit :=
  match name? with
  | some name =>
      Fitree.trigger (SSAEnvE.Set τ name v)
  | none =>
      return ()

-- Handlers

@[simp_itree]
private def stateT_defaultHandler: E ~> StateT (SSAEnv δ) (Fitree E) :=
  fun _ e => StateT.lift $ Fitree.trigger e

@[simp_itree]
private def writerT_defaultHandler: E ~> WriterT (StateT (SSAEnv δ) (Fitree E)) :=
  fun _ e => WriterT.lift $ StateT.lift $ Fitree.trigger e

def interp_ssa {E R} (t: Fitree (SSAEnvE δ +' E) R):
    StateT (SSAEnv δ) (Fitree E) R :=
  interp_state (case_ SSAEnvE.handle stateT_defaultHandler) t

def interp_ssa_logged {E R} (t: Fitree (SSAEnvE δ +' E) R):
    WriterT (StateT (SSAEnv δ) (Fitree E)) R :=
  interp_writer (case_ SSAEnvE.handleLogged writerT_defaultHandler) t
