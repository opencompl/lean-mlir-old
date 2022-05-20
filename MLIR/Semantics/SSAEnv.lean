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

-- | An SSA environment with a single empty SSAScope
def SSAEnv.empty: SSAEnv := [[]]

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

inductive SSAEnvE: Type u → Type _ where
  | Get: (τ: MLIRTy) → [Inhabited τ.eval] → SSAVal → SSAEnvE (ULift τ.eval)
  | Set: (τ: MLIRTy) → SSAVal → τ.eval → SSAEnvE PUnit

@[simp_itree]
def SSAEnvE.handle {E}: SSAEnvE ~> StateT SSAEnv (Fitree E) :=
  fun _ e env =>
    match e with
    | Get τ name =>
        match env.get name τ with
        | some v => return (ULift.up v, env)
        | none => return (ULift.up default, env)
    | Set τ name v =>
        return (.unit, env.set name τ v)

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

structure WriterT (m: Type _ -> Type _) (a: Type _) where
  val: m (a × String)

def WriterT.run (wm: WriterT m a ): m (a × String) := wm.val

instance [Functor m]: Functor (WriterT m) where
  map f w := { val := Functor.map (fun (a, log) => (f a, log)) w.val }

instance [Pure m]: Pure (WriterT m) where
  pure x := { val := pure (x, "") }

instance [Monad m]: Seq (WriterT m) where
   seq mx my :=
     { val := do
        let wx <- mx.val
        let wy <- (my ()).val
        let wb := wx.fst wy.fst
        return (wb, wx.snd ++ wy.snd) }

instance [Monad m] : SeqLeft (WriterT m) where
   seqLeft mx my :=
     { val := do
        let wx <- mx.val
        let wy <- (my ()).val
        return (wx.fst, wx.snd ++ wy.snd) }

instance [Monad m] : SeqRight (WriterT m) where
   seqRight mx my :=
     { val := do
        let wx <- mx.val
        let wy <- (my ()).val
        return (wy.fst, wx.snd  ++ wy.snd ) }

instance [Bind m] [Pure m]: Bind (WriterT m) where
  bind wma a2wmb :=
    let v := do
      let (va, loga) <- wma.val
      let wb <- (a2wmb va).val
      let (vb, logb) := wb
      return (vb, loga ++ logb)
    { val := v }

def WriterT.lift [Monad m] {α : Type u} (ma: m α): WriterT m α :=
  { val := do let a <- ma; return (a, "") }


instance [Monad m]: MonadLift m (WriterT m) where
  monadLift := WriterT.lift

instance : MonadFunctor m (WriterT m) := ⟨fun f mx => { val := f (mx.val) } ⟩

instance [Monad m] : Applicative (WriterT m) where
  pure := Pure.pure
  seqLeft := SeqLeft.seqLeft
  seqRight := SeqRight.seqRight

instance [Monad m]: Monad (WriterT m) where
  pure := Pure.pure
  bind := Bind.bind
  map  := Functor.map

def logWriterT [Monad m] (s: String): WriterT.{u} m PUnit.{u+1} :=
  { val := pure (.unit, s) }



-- |
-- def SSAEnvE.handleLogged {E} {R: Type} (e: SSAEnvE R):  WriterT (StateT SSAEnv (Fitree E)) R := do
def SSAEnvE.handleLogged (E: Type -> Type _) (R: Type): SSAEnvE ~>  WriterT (StateT SSAEnv (Fitree E)) :=
  fun R e => do
    let env <- WriterT.lift $ StateT.get
    match e with
    | Get τ name =>
        logWriterT ("getting " ++ (ToString.toString name))
        match env.get name τ with
        | some v => do
          let debugStr := match τ with | MLIRTy.int _ => toString v | _ => "unk"
          logWriterT ("[val=" ++ debugStr ++ "];  ")
          return ULift.up v
        | none =>
            logWriterT ("[val=none];  ")
           return ULift.up default
    | Set τ name v =>
        logWriterT ("setting " ++ (ToString.toString name))
        let debugStr := match τ with | MLIRTy.int _ => toString v | _ => "unk"
        logWriterT ("[val=" ++ debugStr ++ "];  ")
        WriterT.lift $ StateT.set (env.set name τ v)
        return .unit

-- private def handleUnkLogged {E: Type -> Type _} {R: Type} (e: E R): WriterT (StateT SSAEnv (Fitree E)) R := do
def handleUnkLogged (E: Type -> Type _): E ~> WriterT (StateT SSAEnv (Fitree E)) :=
  fun R e => do
    let r <- @Fitree.trigger E E _ _  e; -- TODO: understand precisely what this means
    return r

-- Interpretation into the writer monad
def interp_writer [Monad M] {E} (h: E ~> WriterT M):
    forall ⦃R⦄, Fitree E R → WriterT M R := interp h

def interp_ssa_logged {E R} (t: Fitree (SSAEnvE +' E) R):
    WriterT (StateT SSAEnv (Fitree E)) R :=
  let x := (case_ (SSAEnvE.handleLogged E R) (handleUnkLogged E))
  let y := interp_writer x t
  y