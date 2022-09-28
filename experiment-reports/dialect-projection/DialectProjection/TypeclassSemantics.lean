/-
In this file, we explore the use of the tagless final style [1]
to encode SSA semantics.

[1] https://okmij.org/ftp/tagless-final/
-/
import Lean
open Lean

namespace WriterT
def WriterT (m: Type _ -> Type _) (a: Type _) := m (a × String)

def WriterT.run (wm: WriterT m a ): m (a × String) := wm

def WriterT.mk (x: m (a × String)): WriterT m a := x

instance [Functor m]: Functor (WriterT m) where
  map f w := Functor.map (f := m) (fun (a, log) => (f a, log)) w

instance [Pure m]: Pure (WriterT m) where
  pure x := pure (f := m) (x, "")

instance [Monad m]: Seq (WriterT m) where
   seq mx my := WriterT.mk do
    let wx <- mx
    let wy <- (my ())
    let wb := wx.fst wy.fst
    return (wb, wx.snd ++ wy.snd)

instance [Monad m] : SeqLeft (WriterT m) where
   seqLeft mx my := WriterT.mk do
    let wx <- mx
    let wy <- (my ())
    return (wx.fst, wx.snd ++ wy.snd)

instance [Monad m] : SeqRight (WriterT m) where
   seqRight mx my := WriterT.mk do
    let wx <- mx
    let wy <- (my ())
    return (wy.fst, wx.snd  ++ wy.snd )

def WriterT.bindCont [Bind m] [Pure m] (k: α → WriterT m β) (x: α × String):
    WriterT m β := WriterT.mk do
  let y ← k x.fst
  return (y.fst, x.snd ++ y.snd)

def WriterT.bind [Bind m] [Pure m] (wma: WriterT m α) (a2wmb: α → WriterT m β):
    WriterT m β :=
  WriterT.mk do
    let x <- wma
    WriterT.bindCont a2wmb x

instance [Bind m] [Pure m]: Bind (WriterT m) where
  bind wma a2wmb := WriterT.bind wma a2wmb

def WriterT.lift [Monad m] {α : Type u} (ma: m α): WriterT m α :=
  Bind.bind (m := m) ma (fun a => return (a, ""))

instance [Monad m]: MonadLift m (WriterT m) where
  monadLift := WriterT.lift

instance : MonadFunctor m (WriterT m) where
  monadMap f := f

instance [Monad m] : Applicative (WriterT m) where
  pure := Pure.pure
  seqLeft := SeqLeft.seqLeft
  seqRight := SeqRight.seqRight

instance [Monad m]: Monad (WriterT m) where
  pure := Pure.pure
  bind := Bind.bind
  map  := Functor.map

def logWriterT [Monad m] (s: String): WriterT.{u} m PUnit.{u+1} :=
  pure (f := m) (.unit, s)

end WriterT

namespace Fitree
open WriterT
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

def Sum.cases {α β γ} (fα: α → γ) (fβ: β → γ): (α ⊕ β) → γ
  | .inl a => fα a
  | .inr b => fβ b

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

def Fitree.ret {E R}: R → Fitree E R :=
  Fitree.Ret

def Fitree.trigger {E: Type → Type} {F: Type → Type} {T} [Member E F]
    (e: E T): Fitree F T :=
  Fitree.Vis (Member.inject _ e) Fitree.ret


def Fitree.bind {E R T} (t: Fitree E T) (k: T → Fitree E R) :=
  match t with
  | Ret r => k r
  | Vis e k' => Vis e (fun r => bind (k' r) k)

instance {E}: Monad (Fitree E) where
  pure := Fitree.ret
  bind := Fitree.bind

-- Since we only use finite ITrees, we can actually run them when they're
-- fully interpreted (which leaves only the Ret constructor)
def Fitree.run {R}: Fitree Void1 R → R
  | Ret r => r
  | Vis e _ => nomatch e

@[simp] theorem Fitree.run_ret:
  Fitree.run (Fitree.ret r) = r := rfl

def Fitree.translate {E F R} (f: E ~> F): Fitree E R → Fitree F R
  | Ret r => Ret r
  | Vis e k => Vis (f _ e) (fun r => translate f (k r))

@[simp] theorem Fitree.translate_ret:
  Fitree.translate f (Fitree.ret r) = Fitree.ret r := rfl
@[simp] theorem Fitree.translate_vis:
    Fitree.translate f (Vis e k) = Vis (f _ e) (fun r => translate f (k r)) :=
  rfl

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

def Fitree.interp {M} [Monad M] {E} (h: E ~> M) {R}: Fitree E R → M R
  | .Ret r => pure r
  | .Vis e k => Bind.bind (h _ e) (fun t => interp h (k t))

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

def Fitree.interpState {M S} [Monad M] {E} (h: E ~> StateT S M):
    forall {R}, Fitree E R → StateT S M R :=
  interp h

def Fitree.interpWriter {M} [Monad M] {E} (h: E ~> WriterT M):
    forall {R}, Fitree E R → WriterT M R :=
  interp h

def Fitree.interpOption {M} [Monad M] {E} (h: E ~> OptionT M):
    forall {R}, Fitree E R → OptionT M R :=
  interp h

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

@[simp] theorem Fitree.interp_Ret:
  Fitree.interp h (Fitree.Ret r) = Fitree.ret r := rfl

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


end Fitree

namespace Exp


-- https://okmij.org/ftp/tagless-final/course/lecture.pdf
inductive Exp where
| Lit: Int -> Exp
| Neg: Exp -> Exp
| Add: Exp -> Exp -> Exp

def Exp.eval: Exp -> Int
| .Lit i => i
| .Neg e => -1 * e.eval
| .Add e e' => e.eval + e'.eval

class ExpSYM (repr: Type) where
  lit: Int -> repr
  neg: repr -> repr
  add: repr -> repr -> repr
  -- neg_involutive: (a: repr) -> neg (neg a) = a

instance : ExpSYM Int where
  lit i := i
  neg i := (-i)
  add i j := i + j

instance : ExpSYM String where
  lit i := toString i
  neg i := s!"(neg {i})"
  add i i' := s!"(add {i} {i'})"
end Exp

namespace Tree
inductive Tree where
| Leaf: String -> Tree
| Node: String -> List Tree -> Tree
deriving BEq

open Exp

-- Serialize Exp into Tree

instance : ExpSYM Tree where
  lit n := .Node "Lit" [.Leaf (toString n)]
  neg e := .Node "Neg" [e]
  add e e' := .Node "Add" [e, e']


def fromTree {repr: Type} [ExpSYM repr] : Tree -> Except String repr
| .Node "Lit" [.Leaf n] => do
   Except.ok (ExpSYM.lit 42) -- TODO: convert from string to nat.
| .Node "Neg" [e] => do
       return (ExpSYM.neg (<- fromTree e))
| .Node "Add" [e, e'] => do
   return ExpSYM.add (<- fromTree e) (<-
fromTree e')
| _t => Except.error "incorrect tree"

end Tree

namespace PushNeg
open Exp

def Exp.pushNeg: Exp -> Exp
| .Lit v => .Lit v
| .Neg (.Lit v) => .Neg (.Lit v)
| .Neg (.Neg e) => Exp.pushNeg e
| .Neg (.Add e e') => .Add (Exp.pushNeg e) (Exp.pushNeg e')
| .Add e e' => .Add (Exp.pushNeg e) (Exp.pushNeg e')

inductive Ctx where
| Pos: Ctx
| Neg: Ctx

instance {repr: Type} [ExpSYM repr] : ExpSYM (Ctx -> repr) where
  lit n := fun ctx => match ctx with
    | .Pos => ExpSYM.lit n
    | .Neg => ExpSYM.neg (ExpSYM.lit n)
  neg e := fun ctx => match ctx with
    | .Pos => e .Neg
    | .Neg => e .Pos
  add e1 e2 := fun ctx =>  ExpSYM.add (e1 ctx) (e2 ctx)
end PushNeg

namespace HO -- higher order tagless final

class Symantics (repr: Type -> Type) where
  int : Int -> repr Int
  add: repr Int -> repr Int -> repr Int
  lam: (repr a -> repr b) -> repr (a -> b)
  app: repr (a -> b) -> repr a -> repr b

structure R (a: Type) where
  val : a

instance : Symantics R where
  int i := { val := i }
  add i j := { val := i.val + j.val }
  lam f := { val := fun a =>  (f (R.mk a)).val }
  app f a := R.mk $ f.val a.val

class BoolSYM (repr: Type -> Type) where
  bool: Bool -> repr Bool
  leq : repr Int -> repr Int -> repr Bool
  if_: repr Bool -> repr a -> repr a -> repr a

instance : BoolSYM R where
 bool b := R.mk b
 leq a a' := R.mk (a.val <= a'.val)
 if_ cond t e := R.mk $ if cond.val then t.val else e.val

class FixSYM (repr: Type -> Type) where
  fix: (repr a -> repr a) -> repr a

-- lol
partial instance : FixSYM R where
  fix := sorry


-- h is heaps

inductive IR (h: Type _ -> Type _): Type _ -> Type _ where
| int: Int -> IR h Int
| add: IR h t -> IR h t -> IR h t
| var: h t -> IR h t
-- | lam: (IR h t1 -> IR h t2) -> IR h (t1 -> t2) -- non-positive occurence, cannot be encoded in initial style!


end HO



namespace SSA
/-
inductive BB (repr: Type _ -> Type _ ): Type _ -> Type _ where
| entry: String -> BB repr a -> BB repr a -- begin a bb
| seq: BB repr a -> BB repr b -> BB repr b
| op: repr a -> BB repr a -- operation
| ret: repr a -> BB repr a -- only place where problem occurs.
| condbr: repr Bool -> String -> String -> BB repr Unit
| br: String -> BB repr Unit


class BBSemantics (repr: Type _ -> Type _) where
  bb: BB repr a -> repr a

structure R (a: Type) where
  val : a


instance : BBSemantics R where
  bb repr := match repr with
             | .entry name rest =>

-/

inductive Op: Type _ -> Type _ where
| add: Int -> Int -> Op Int
| lt: Int -> Int -> Op Bool
| const: Int -> Op Int

class OpSYM (repr: Type -> Type) where
  add: Int -> Int -> repr Int
  lt: Int -> Int -> repr Bool
  const: Int -> repr Int


instance : OpSYM Op where
  add := .add
  lt := .lt
  const := .const

structure BBName where
  name: String

structure BBRef (a: Type _) where
  name: String

-- class BBRefSYM (repr: Type -> Type) := String -> repr a

-- Terminator has single type for interprocedural control flow.
-- Inside and Outside
-- k for things that are unknown, in the grand CPS style
-- BB intra inter.
inductive Terminator: Type _ -> Type _ where
| br: BBRef i -> i -> Terminator Unit
| ret: o -> Terminator o
| condbr: Bool -> (BBRef i × i) -> (BBRef i' × i') -> Terminator Unit

class TerminatorSYM (repr: Type _ -> Type _) where
  br: BBRef i -> i -> repr Unit
  ret: o -> repr o
  condbr: Bool -> (BBRef i × i) -> (BBRef i' × i') -> repr Unit

instance : TerminatorSYM Terminator where
  br := .br
  ret := .ret
  condbr := .condbr

-- BB has three two type: one for interprocedural control flow
-- one for intraprocedural control flow
-- Inside and Outside
-- BB intra inter.
-- BB <input-type> <interprocedural-out-type>
-- O: type of ops
-- T: type of terminators.
inductive BB (O: Type _ -> Type _) (T: Type _ -> Type _): Type _ -> Type _ -> Type _ where
| begin: (i -> BB O T Unit o) -> BB O T i o
| seq: O a -> (a -> BB O T Unit o) -> BB O T Unit o
| terminator: T o -> BB O T Unit o

class  BBSYM (bbRepr: Type _ -> Type _ -> Type _)
  (opRepr: Type _ -> Type _)
  (terminatorRepr: Type _ -> Type _)
  extends OpSYM opRepr, TerminatorSYM terminatorRepr where
  begin: (i -> bbRepr Unit o) -> bbRepr i o
  seq: (opRepr a) -> (a -> bbRepr Unit o) -> bbRepr Unit o
  terminator: (terminatorRepr o) -> bbRepr Unit o

-- instance of Symantics for BB.
instance [OpSYM O] [TerminatorSYM T]: BBSYM (BB O T) O T where
  begin := BB.begin
  seq := BB.seq
  terminator := BB.terminator


-- build a BB which takes 'Int' input, produces 'Int' output.
def prog0 : BB Op Terminator Int Int :=
  .begin (fun input =>
    .seq (.const 4) (fun j =>
    .seq (.add input j) (fun k =>
    .terminator (.ret k)
  )))


namespace RegionBuilder
-- Build a region
-- The list of types is the labels that have been defined.
inductive RegionBuilder
  (O: Type _ -> Type _)
  (T: Type _ -> Type _): List (Σ (i: Type), BBRef i) -> Type _ -> Type _ -> Type _ where
| lbl: ((ref: BBRef i) ->
   RegionBuilder O T (⟨ i, ref ⟩::ris) ri ro) -- if you want a label,
   ->  RegionBuilder O T ris ri ro -- I can then forget about the `i` and remember that the `is` have been defined
                                   -- you have an obligation to define it in the output
| define: (ref: BBRef i) -> BB O T i o -> RegionBuilder O T ris ri ro
    -> RegionBuilder O T (⟨i,ref⟩::ris) ri ro -- define defines an `i`.
| empty: RegionBuilder O T [] ri ro -- empty region defines no BBS.

def prog1: RegionBuilder Op Terminator [] Int Int :=
  .lbl (i := Int) (fun entry =>
     .define entry (.begin fun i =>
      .terminator (.ret i)
      ) .empty)

-- takes an int as input, produces an int as output
-- entry(input):
--   br loop (input, 0)
-- loop(i, k):
--   knew := k + 1
--   inew := i + 1
--   exit := knew == 10
--   condbr exit(inew), loop(inew, knew)
-- exit(inew):
--   ret inew
def prog2: RegionBuilder Op Terminator [] Int Int :=
  .lbl (i := Int) (fun entrybb =>
  .lbl (i := Int × Int) (fun loopbb =>
  .lbl (i := Int) (fun exitbb =>
     .define exitbb (.begin fun inew => .terminator (.ret inew)) $
     .define loopbb (.begin fun args =>
       .seq (.add 1 args.fst) (fun knew =>
       .seq (.add 1 args.snd) (fun inew =>
       .seq (.lt knew 10) (fun isExit =>
       -- .terminator (.ret knew)))) -- (.condbr isExit ⟨exitbb, inew⟩, ⟨loopbb, (inew, knew)⟩)))))
       .terminator (.condbr isExit (exitbb, inew) (loopbb, (inew, knew))))))
     ) $
     .define entrybb (.begin fun input =>
      .terminator (.br loopbb (input, 0))) $
     .empty)))
#reduce prog2
end RegionBuilder

namespace Region

end Region

end SSA

namespace StructuredSSA
/-
We flatten Op, BasicBlock, Region into a single Def'.
We need three notions:
(1) Running some semantic value (R), labelled by a label (L)
(3) creating a new scope
(2) invoking control flow (C) to a label (L)
(2) sequentially composing two defs

-/

inductive Producer: Type -> Type where
inductive Consumer: Type -> Type where
inductive ProducerConsumer: Type -> Type -> Type where


-- op: dataflow
-- bb: ? (Ill defined concept)
-- br, condbr: control flow.
-- CFG: control flow

-- backwards dataflow graph.
inductive Dataflow (D: Type -> Type -> Type) (C: Type -> Type -> Type): Type _ -> Type _ where
| val: O -> Dataflow D C O
| df: D I O -> (I -> Dataflow D C O) -> Dataflow D C O

-- forwards control flow graph.
inductive Controlflow (D: Type -> Type -> Type) (C: Type -> Type -> Type): Type _ -> Type _ where
| controldep: (I -> Dataflow D C O) -> Controlflow D C I -- Create phi nodes / control flow dependent values.
| cf: C I BLANK  /- instruction condbr in conbr b bb1, bb2 (I = Bool) -/
   -- -> (I -> Dataflow D C O' × Controlflow D C I') /- function that maps true ->bb1(x), false -> bb2(x), and shows how to produce (x, y) when mapping. -/
   -> (I ->  Controlflow D C I') /- function that maps true ->bb1, false -> bb2 -/

inductive Void where

abbrev Unit2 (a: Type) := a
abbrev Void2 (_a: Type) := Void



inductive OpD : Type -> Type where -- tagged by output type
| add: Int -> Int -> OpD Int
| neg: Int -> OpD Int

abbrev Op O := Dataflow OpD Void2 O

inductive TerminatorC : Type -> Type where -- tagged by input type
|  br: TerminatorC Unit
|  condbr: TerminatorC Bool

-- A basic block is obtained by taking the data flow of an Op and the control flow of a Terminator
abbrev BasicBlock I := Controlflow OpD TerminatorC I

inductive Adapt (D: Type -> Type _) (C: Type -> Type _): Type _ -> Type _
| adapt: D O ->  C I -> (O -> I) -> Adapt D C O

-- A region adapts
abbrev Region O := Adapt BasicBlock  BasicBlock O

end StructuredSSA

namespace PartialFunctionReasoning

-- @[mlirdBy "factorial"]
opaque factorial: Int -> Int
-- axiom factorial_succ: ∀ (n: Nat), factorial (Int.ofNat (Nat.succ n))
axiom factorial_rec: ∀ (i: Nat), factorial (Int.ofNat (Nat.succ i)) = (Nat.succ i) * factorial (Int.ofNat i)
axiom factorial_zero: factorial (Int.ofNat 0) = 1

def terminating_factorial (n: Nat): Nat :=
  match n with
  | 0 => 1
  | n' + 1 => n * terminating_factorial n'

#check Nat
theorem agree: forall (n: Nat), terminating_factorial n = factorial (.ofNat n) := by {
  intros n;
  induction n;
  case zero =>  {
  simp [factorial_zero, terminating_factorial];
  }
  case succ n H => {
   simp[terminating_factorial, H];
   simp[factorial_rec];
   rewrite [<- H];
   sorry
  }

}

end PartialFunctionReasoning

namespace PartialEvaluator
-- Section 4.6
end PartialEvaluator
