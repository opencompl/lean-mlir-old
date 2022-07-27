/-
## Undefined Behavior

This file models undefined behavior, the runtime condition of reaching a state
for which semantics are undefined. This captures errors that are dependent on
runtime data, eg. division by zero.


Undefined behavior is interpreted in the exception monad, with generic string-
based error messages. An execution that runs into UB is stopped; determinism
is preserved.

TODO: Consider an UB interpreter directly into results from a proof of non-UB
-/

import MLIR.Semantics.Fitree

-- The monad for UB records exceptions based on strings
abbrev UBT := ExceptT String

inductive UBE: Type → Type :=
  | UB {α: Type} [Inhabited α]: Option String → UBE α

@[simp_itree]
def UBE.handle {E}: UBE ~> UBT (Fitree E) := fun _ e =>
  match e with
  | UB none => throw "<UB>"
  | UB (some msg) => throw s!"<UB: {msg}>"

@[simp_itree]
def UBE.handle! {E}: UBE ~> Fitree E := fun _ e =>
  match e with
  | UB none => panic! "<UB>"
  | UB (some msg) => panic! s!"<UB: {msg}>"

def raiseUB (msg: String) {E α} [Member UBE E] [Inhabited α]: Fitree E α :=
  Fitree.trigger <| UBE.UB (some msg)

def interpUB (t: Fitree UBE R): UBT (Fitree Void1) R :=
  t.interpExcept UBE.handle

def interpUB! (t: Fitree UBE R): Fitree Void1 R :=
  t.interp UBE.handle!

def interpUB' {E} (t: Fitree (UBE +' E) R): UBT (Fitree E) R :=
  t.interpExcept (Fitree.case UBE.handle Fitree.liftHandler)

def interpUB'! {E} (t: Fitree (UBE +' E) R): Fitree E R :=
  t.interp (Fitree.case UBE.handle! (fun T => @Fitree.trigger E E T _))

@[simp] theorem interpUB'_Vis_right:
  interpUB' (Fitree.Vis (Sum.inr e) k) =
  Fitree.Vis e (fun x => interpUB' (k x)) := rfl

@[simp] theorem interpUB'_ret:
  @interpUB' _ E (Fitree.ret r) = Fitree.ret (Except.ok r) := rfl

theorem interpUB'_bind (k: T → Fitree (UBE +' E) R):
  interpUB' (Fitree.bind t k) =
  Fitree.bind (interpUB' t) (fun x =>
    match x with
    | .error ε => Fitree.ret (.error ε)
    | .ok x => interpUB' (k x)) := by
  -- Can't reuse `Fitree.interpExcept_bind` because the match statements are
  -- considered different by isDefEq for some reason
  induction t with
  | Ret _ => rfl
  | Vis _ _ ih =>
      simp [interpUB', Fitree.interpExcept] at *
      simp [Fitree.interp, Fitree.bind, Bind.bind]
      simp [ExceptT.bind, ExceptT.mk, ExceptT.bindCont]
      have fequal2 α β (f g: α → β) x y: f = g → x = y → f x = g y :=
        fun h₁ h₂ => by simp [h₁, h₂]
      apply fequal2; rfl; funext x
      cases x <;> simp [ih]
