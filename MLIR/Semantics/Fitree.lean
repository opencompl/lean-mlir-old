/-
## Finite interaction trees

The semantics framework for this project is extremely inspired by the Vellvm
project [1] and is essentially centered around interaction trees and monadic
transformers.

Interactions trees are a particular instance of the freer monad; essentially,
an ITree is a program that can have side effets through *interactions*, and
these interactions can either be interpreted into the program or kept as
observable side-effects.

When giving semantics to a program, one usually starts with a rather simple
ITree where most of the complex features of the language (memory, I/O,
exceptions, randomness, non-determinism, etc) are hidden behind interactions.
The interactions are then interpreted, which consists of (1) enriching the
program's environment by a monadic transformation, and (2) replacing the
interaction with an actual implementation.

This approach allows monadic domains to be used while keeping each family of
interactions separate. This is relevant for Vellvm as LLVM IR has many complex
features, and even more relevant for MLIR since each dialect can bring more
interactions and environment transforms and all of them have to be studied and
defined independently.

The datatype of interaction trees normally has built-in non-termination by
being defined coinductively. Support for coinduction is still limited in
Lean 4, so we currently use a finite version of ITrees (hence called Fitree)
and we only model programs that always terminate.

[1]: https://github.com/vellvm/vellvm
-/

import MLIR.Semantics.SimpItree
import MLIR.Dialects
import MLIR.Util.WriterT

/- Extendable effect families -/

abbrev to1 (E: Type → Type u) (F: Type → Type v) :=
  ∀ T, E T → F T
abbrev sum1 (E F: Type → Type) :=
  fun T => E T ⊕ F T
inductive Void1: Type → Type :=

infixr:40 " ~> " => to1
infixr:60 " +' " => sum1

class Member (E: Type → Type) (F: Type → Type) where
  inject : E ~> F

instance MemberId {E}: Member E E where
  inject := (fun _ => id)

instance MemberSumL {E F G} [Member E F]: Member E (F +' G) where
  inject T := Sum.inl ∘ Member.inject T

instance MemberSumR {E F G} [Member E G]: Member E (F +' G) where
  inject T := Sum.inr ∘ Member.inject T

instance MemberSum {E F G H} [Member E G] [Member F H]:
    Member (E +' F) (G +' H) where
  inject T := Sum.cases (Member.inject T) (Member.inject T)

instance MemberVoid1 {E}:
    Member Void1 E where
  inject _ e := nomatch e

-- Effects can now be put in context automatically by typeclass resolution
example E:      Member E E := inferInstance
example E F:    Member E (E +' F) := inferInstance
example E F:    Member E (F +' (F +' E)) := inferInstance
example E F G:  Member (E +' F) (E +' F +' G) := inferInstance


/- The monadic domain; essentially finite Interaction Trees -/

inductive Fitree (E: Type → Type) (R: Type) where
  | Ret (r: R): Fitree E R
  | Vis {T: Type} (e: E T) (k: T → Fitree E R): Fitree E R

@[simp_itree]
def Fitree.ret {E R}: R → Fitree E R :=
  Fitree.Ret

@[simp_itree]
def Fitree.trigger {E: Type → Type} {F: Type → Type} {T} [Member E F]
    (e: E T): Fitree F T :=
  Fitree.Vis (Member.inject _ e) Fitree.ret


@[simp_itree]
def Fitree.bind {E R T} (t: Fitree E T) (k: T → Fitree E R) :=
  match t with
  | Ret r => k r
  | Vis e k' => Vis e (fun r => bind (k' r) k)

instance {E}: Monad (Fitree E) where
  pure := Fitree.ret
  bind := Fitree.bind

-- Since we only use finite ITrees, we can actually run them when they're
-- fully interpreted (which leaves only the Ret constructor)
@[simp_itree]
def Fitree.run {R}: Fitree Void1 R → R
  | Ret r => r
  | Vis e _ => nomatch e

@[simp] theorem Fitree.run_ret:
  Fitree.run (Fitree.ret r) = r := rfl

@[simp_itree]
def Fitree.translate {E F R} (f: E ~> F): Fitree E R → Fitree F R
  | Ret r => Ret r
  | Vis e k => Vis (f _ e) (fun r => translate f (k r))

@[simp] theorem Fitree.translate_ret:
  Fitree.translate f (Fitree.ret r) = Fitree.ret r := rfl
@[simp] theorem Fitree.translate_vis:
    Fitree.translate f (Vis e k) = Vis (f _ e) (fun r => translate f (k r)) :=
  rfl

@[simp_itree]
def Fitree.case (h₁: E ~> G) (h₂: F ~> G): E +' F ~> G :=
  fun R ef => match ef with
  | Sum.inl e => h₁ R e
  | Sum.inr f => h₂ R f

@[simp] theorem Fitree.case_left:
  Fitree.case h₁ h₂ _ (Sum.inl e) = h₁ _ e := rfl
@[simp] theorem Fitree.case_right:
  Fitree.case h₁ h₂ _ (Sum.inr e) = h₂ _ e := rfl

/-
### Monadic interpretation
-/

@[simp_itree]
def Fitree.interp {M} [Monad M] {E} (h: E ~> M) {R}: Fitree E R → M R
  | .Ret r => pure r
  | .Vis e k => Bind.bind (h _ e) (fun t => interp h (k t))

@[simp_itree]
def Fitree.interp' {E F} (h: E ~> Fitree Void1) {R} (t: Fitree (E +' F) R):
    Fitree F R :=
  interp (Fitree.case
    (fun _ e => (h _ e).translate $ fun _ e => nomatch e)
    (fun _ e => Fitree.trigger e)) t

-- Interp `F` by lifting into a monad transformer (this is used when
-- interpreting `E +' F` into the monad)
def Fitree.liftHandler {F M} [MonadLiftT (Fitree F) M]: F ~> M := fun R e =>
  monadLift (Fitree.trigger e: Fitree F R)

-- Interpretation into various predefined monads. These are predefined so that
-- rewriting theorems that expose the monad structure can be provided.

@[simp_itree]
def Fitree.interpState {M S} [Monad M] {E} (h: E ~> StateT S M):
    forall {R}, Fitree E R → StateT S M R :=
  interp h

@[simp_itree]
def Fitree.interpWriter {M} [Monad M] {E} (h: E ~> WriterT M):
    forall {R}, Fitree E R → WriterT M R :=
  interp h

@[simp_itree]
def Fitree.interpOption {M} [Monad M] {E} (h: E ~> OptionT M):
    forall {R}, Fitree E R → OptionT M R :=
  interp h

@[simp_itree]
def Fitree.interpExcept {M ε} [Monad M] {E} (h: E ~> ExceptT ε M) {R}:
    Fitree E R → ExceptT ε M R :=
  interp h

/-
### Combinator identities

The following theorems act as the main interface for computation on ITrees. We
don't unfold definitions because Lean 4 doesn't yet have the match-unfolding
behavior of Coq's `simpl` tactic, and runs into performance issues as unfolded
terms grow larger. Instead, we aggressively rewrite the following simplifying
equalities.
-/

@[simp] theorem Fitree.bind_ret:
  Fitree.bind (Fitree.ret r) k = k r := rfl

@[simp] theorem Fitree.bind_Ret:
  Fitree.bind (Fitree.Ret r) k = k r := rfl

@[simp] theorem Fitree.bind_ret':
    Fitree.bind t (fun r => Fitree.ret r) = t := by
  induction t with
  | Ret _ => rfl
  | Vis _ _ ih => simp [bind, ih]

@[simp] theorem Fitree.bind_Ret':
    Fitree.bind t (fun r => Fitree.Ret r) = t := by
  induction t with
  | Ret _ => rfl
  | Vis _ _ ih => simp [bind, ih]

@[simp] theorem Fitree.bind_bind:
    Fitree.bind (Fitree.bind t k) k' =
    Fitree.bind t (fun x => Fitree.bind (k x) k') := by
  induction t with
  | Ret _ => rfl
  | Vis _ _ ih => simp [bind, ih]

@[simp] theorem Fitree.pure_is_ret:
  @Pure.pure (Fitree E) _ _ r = Fitree.ret r := rfl

@[simp] theorem Fitree.bind_is_bind:
  @Bind.bind (Fitree E) _ _ _  t k = Fitree.bind t k := rfl

@[simp] theorem Fitree.StateT_bind_is_bind (k: T → S → Fitree E (R × S)):
  StateT.bind (m := Fitree E) t k =
    fun s => Fitree.bind (t s) (fun (x,s) => k x s) := rfl

@[simp] theorem Fitree.WriterT_bind_is_bind (k: T → Fitree E (R × String)):
  WriterT.bind (m := Fitree E) t k =
    Fitree.bind t (WriterT.bindCont k) := rfl

@[simp] theorem Fitree.OptionT_bind_is_bind (k: T → Fitree E (Option R)):
  OptionT.bind (m := Fitree E) t k =
    Fitree.bind t (fun
      | some x => k x
      | none => Fitree.ret none) := rfl

@[simp] theorem Fitree.ExceptT_bind_is_bind (k: T → Fitree E (Except ε R)):
  ExceptT.bind (m := Fitree E) t k = Fitree.bind t (ExceptT.bindCont k) := rfl

@[simp] theorem Fitree.liftHandler_StateT_is_StateT_lift:
  @Fitree.liftHandler F (StateT S (Fitree F)) _ _ e =
  fun s => Fitree.bind (Fitree.trigger e) (fun x => Fitree.ret (x, s)) := rfl

@[simp] theorem Fitree.liftHandler_WriterT_is_WriterT_lift:
  @Fitree.liftHandler F (WriterT (Fitree F)) _ _ e =
  Fitree.bind (Fitree.trigger e) (fun x => Fitree.ret (x, "")) := rfl

@[simp] theorem Fitree.liftHandler_OptionT_is_OptionT_lift:
  @Fitree.liftHandler F (OptionT (Fitree F)) _ _ e =
  Fitree.bind (Fitree.trigger e) (fun x => Fitree.ret (some x)) := rfl

@[simp] theorem Fitree.liftHandler_ExceptT_is_ExceptT_lift:
  @Fitree.liftHandler F (ExceptT ε (Fitree F)) _ _ e =
  Fitree.bind (Fitree.trigger e) (fun x => Fitree.ret (Except.ok x)) := rfl

@[simp] theorem Member.injectId:
  @Member.inject E E MemberId _ e = e := rfl

@[simp] theorem Member.injectSumL [Member E F]:
  @Member.inject E (F +' G) MemberSumL _ e = Sum.inl (Member.inject _ e) := rfl

@[simp] theorem Member.injectSumR [Member E G]:
  @Member.inject E (F +' G) MemberSumR _ e = Sum.inr (Member.inject _ e) := rfl

@[simp] theorem Member.injectSum_inl [Member E G] [Member F H]:
  @Member.inject (E +' F) (G +' H) MemberSum _ (Sum.inl e) =
    Sum.inl (Member.inject _ e) := rfl

@[simp] theorem Member.injectSum_inr [Member E G] [Member F H]:
  @Member.inject (E +' F) (G +' H) MemberSum _ (Sum.inr e) =
    Sum.inr (Member.inject _ e) := rfl

-- Interpretatin identities


@[simp] theorem Fitree.interp_ret:
  Fitree.interp h (Fitree.ret r) = Fitree.ret r := rfl

@[simp] theorem Fitree.interp_Vis:
  Fitree.interp h (Fitree.Vis e k) =
  Fitree.bind (h _ e) (fun x => Fitree.interp h (k x)) := rfl

@[simp] theorem Fitree.interp'_ret:
  @Fitree.interp' E F h _ (Fitree.ret r) = Fitree.ret r := rfl

@[simp] theorem Fitree.interp'_Vis_left:
  @Fitree.interp' E F h _ (Fitree.Vis (Sum.inl e) k) =
  Fitree.bind (Fitree.translate (fun _ e => nomatch e) (h _ e))
              (fun x => Fitree.interp' h (k x)) := rfl

@[simp] theorem Fitree.interp'_Vis_right:
  @Fitree.interp' E F h _ (Fitree.Vis (Sum.inr e) k) =
  Fitree.bind (Fitree.trigger e)
              (fun x => Fitree.interp' h (k x)) := rfl

@[simp] theorem Fitree.interpState_ret:
  Fitree.interpState h (Fitree.ret r) = (fun s => Fitree.ret (r, s)) := rfl

@[simp] theorem Fitree.interpState_Vis {M S} [Monad M] (h: E ~> StateT S M):
  Fitree.interpState h (Fitree.Vis e k) =
  StateT.bind (h _ e) (fun x => Fitree.interpState h (k x)) := rfl

@[simp] theorem Fitree.interpWriter_ret:
  Fitree.interpWriter h (Fitree.ret r) = Fitree.ret (r, "") := rfl

@[simp] theorem Fitree.interpWriter_Vis {M} [Monad M] (h: E ~> WriterT M):
  Fitree.interpWriter h (Fitree.Vis e k) =
  WriterT.bind (h _ e) (fun x => Fitree.interpWriter h (k x)) := rfl

@[simp] theorem Fitree.interpOption_ret:
  Fitree.interpOption h (Fitree.ret r) = Fitree.ret (some r) := rfl

@[simp] theorem Fitree.interpOption_Vis {M} [Monad M] (h: E ~> OptionT M):
  Fitree.interpOption h (Fitree.Vis e k) =
  OptionT.bind (h _ e) (fun x => Fitree.interpOption h (k x)) := rfl

@[simp] theorem Fitree.interpExcept_ret:
  Fitree.interpExcept h (Fitree.ret r) = Fitree.ret (.ok r) := rfl

@[simp] theorem Fitree.interpExcept_Vis {M ε} [Monad M] (h: E ~> ExceptT ε M):
  Fitree.interpExcept h (Fitree.Vis e k) =
  ExceptT.bind (h _ e) (fun x => Fitree.interpExcept h (k x)) := rfl

-- We don't assume [LawfulMonad M] so we can't simplify the continuation. But
-- when it's an ITree the other simp lemmas will do it anyway.
@[simp] theorem Fitree.interp_trigger [Member E F] [Monad M] (e: E T):
  Fitree.interp (M := M) (E := F) h (Fitree.trigger e) =
  Bind.bind (h _ (Member.inject _ e)) (fun x => pure x) := rfl

@[simp] theorem Fitree.interp'_trigger_left (e: E R):
  @Fitree.interp' E F h _ (@Fitree.trigger (E +' F) _ _ MemberId (Sum.inl e)) =
  Fitree.bind
    (Fitree.translate (fun _ e => nomatch e) (h _ (Member.inject _ e)))
    (fun x => pure x) := rfl

@[simp] theorem Fitree.interp'_trigger_right [Member G F]:
  @Fitree.interp' E F h _ (@Fitree.trigger (E +' G) (E +' F) _ _ (Sum.inr e)) =
  Fitree.trigger e := rfl

-- The following theorems are only applied manually

theorem Fitree.run_bind {T R} (t: Fitree Void1 T) (k: T → Fitree Void1 R):
    run (bind t k) = run (k (run t)) :=
  match t with
  | Ret _ => rfl
  | Vis e _ => nomatch e

theorem Fitree.interp_bind:
    Fitree.interp h (Fitree.bind t k) =
    Fitree.bind (Fitree.interp h t) (fun x => Fitree.interp h (k x)) := by
  induction t with
  | Ret _ => rfl
  | Vis _ _ ih => simp [bind, ih]

theorem Fitree.interp'_bind:
    Fitree.interp' h (Fitree.bind t k) =
    Fitree.bind (Fitree.interp' h t) (fun x => Fitree.interp' h (k x)) := by
  simp [interp', interp_bind]

-- Specialized interp_bind lemmas that unfold the monadic structure and expose
-- the Fitree.bind directly rather than the monadic Bind.bind

theorem Fitree.interpState_bind (h: E ~> StateT S (Fitree F)) (t: Fitree E R):
    Fitree.interpState h (Fitree.bind t k) s =
    Fitree.bind (Fitree.interpState h t s)
      (fun (x,s') => Fitree.interpState h (k x) s') := by
  revert s
  induction t with
  | Ret _ => intros s; rfl
  | Vis _ _ ih =>
    simp [interpState] at *
    simp [interp, Bind.bind, StateT.bind]
    simp [ih]

example {F R}: WriterT (Fitree F) R = Fitree F (R × String) := by
  simp [WriterT]

theorem Fitree.interpWriter_bind (h: E ~> WriterT (Fitree F))
  (t: Fitree E T) (k: T → Fitree E R):
    Fitree.interpWriter h (Fitree.bind t k) =
    Fitree.bind (Fitree.interpWriter h t) fun (x,s₁) =>
      Fitree.bind (Fitree.interpWriter h (k x)) fun (y,s₂) =>
        Fitree.ret (y,s₁++s₂) := by
  induction t with
  | Ret _ =>
      simp [bind, interpWriter]
      have h₁: forall x, "" ++ x = x := by
        simp [HAppend.hAppend, Append.append, String.append]
        simp [List.nil_append]
      simp [h₁]
      have h₂: forall (α β: Type) (x: α × β), (x.fst, x.snd) = x := by simp
      simp [h₂]
  | Vis _ _ ih =>
      simp [interpWriter] at *
      simp [interp, Bind.bind, WriterT.bindCont, WriterT.mk]
      have h: forall (x y z: String), x ++ (y ++ z) = x ++ y ++ z := by
        simp [HAppend.hAppend, Append.append, String.append]
        simp [List.append_assoc]
      simp [ih, h]

theorem Fitree.interpOption_bind (h: E ~> OptionT (Fitree F))
  (t: Fitree E T) (k: T → Fitree E R):
    Fitree.interpOption h (Fitree.bind t k) =
    Fitree.bind (Fitree.interpOption h t) fun x? =>
      match x? with
      | some x => Fitree.interpOption h (k x)
      | none => Fitree.ret none := by
  induction t with
  | Ret _ => rfl
  | Vis _ _ ih =>
      simp [interpOption] at *
      simp [interp, bind, Bind.bind, OptionT.bind, OptionT.mk]
      -- I can't get a bind (match) → match (bind) theorem to rewrite, so...
      have fequal2 α β (f g: α → β) x y: f = g → x = y → f x = g y :=
        fun h₁ h₂ => by simp [h₁, h₂]
      apply fequal2; rfl; funext x
      cases x <;> simp [ih]

theorem Fitree.interpExcept_bind (h: E ~> ExceptT ε (Fitree F))
  (t: Fitree E T) (k: T → Fitree E R):
    Fitree.interpExcept h (Fitree.bind t k) =
    Fitree.bind (Fitree.interpExcept h t) fun x? =>
      match x? with
      | .error ε => Fitree.ret (.error ε)
      | .ok x => Fitree.interpExcept h (k x) := by
  induction t with
  | Ret _ => rfl
  | Vis _ _ ih =>
      simp [interpExcept] at *
      simp [interp, bind, Bind.bind]
      simp [ExceptT.bind, ExceptT.mk, ExceptT.bindCont]
      -- See above
      have fequal2 α β (f g: α → β) x y: f = g → x = y → f x = g y :=
        fun h₁ h₂ => by simp [h₁, h₂]
      apply fequal2; rfl; funext x
      cases x <;> simp [ih]

-- This theorem has the drawback of hiding the continuation of `bind` into the
-- `Vis` node, which blocks other theorems like `Fitree.bind_bind`.
theorem Fitree.bind_trigger [Member E F] (e: E T) (k: T → Fitree F R):
  Fitree.bind (Fitree.trigger e) k = Fitree.Vis (Member.inject _ e) k := rfl

/-
### Other properties
-/

inductive Fitree.noEventL {E F R}: Fitree (E +' F) R → Prop :=
  | Ret r: noEventL (Ret r)
  | Vis f k: (∀ t, noEventL (k t)) → noEventL (Vis (Sum.inr f) k)


/-
### Automation tools
-/

/- Rewriting tactics `simp_itree` and `dsimp_itree` -/

open Lean Elab.Tactic Parser.Tactic

def toSimpLemma (name: Name): Syntax :=
  mkNode `Lean.Parser.Tactic.simpLemma #[mkNullNode, mkNullNode, mkIdent name]

def tacticSimpItree (definitional: Bool): TacticM Unit := do
  -- TODO: Also handle .lemmaNames, not just unfolding!
  let lemmas := (← SimpItreeExtension.getTheorems).toUnfold.fold
    (init := #[]) (fun acc n => acc.push (toSimpLemma n))
  let others := [
    ``Fitree.liftHandler, ``Member.inject,
    ``StateT.bind, ``StateT.pure, ``StateT.lift,
    ``OptionT.bind, ``OptionT.pure, ``OptionT.mk, ``OptionT.lift,
    ``ExceptT.pure, ``ExceptT.mk,
    ``bind, ``pure, ``cast_eq, ``Eq.mpr].map toSimpLemma
  let fullSet :=
    (lemmas.reverse ++ others).toList.intersperse (mkAtom ",") |>.toArray
  if definitional then
    evalTactic $ ← `(tactic|dsimp [$(⟨fullSet⟩),*])
  else
    evalTactic $ ← `(tactic|simp [$(⟨fullSet⟩),*])

elab "simp_itree":  tactic => tacticSimpItree false
elab "dsimp_itree": tactic => tacticSimpItree true
