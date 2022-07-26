/-
## Undefined Behavior events

This event models undefined behavior. It is emitted in a variety of situations,
such as operations that don't satisfy their invariants, or usual runtime UB
situations such as integer overflow, division by zero, etc. For debug purposes,
it comes with a warning message that is displayed on stderr.

There is another interpreter which eliminates them entirely provided a proof
that the entire program's behavior is defined. (But building that proof is not
practical yet.)
-/

import MLIR.Semantics.Fitree

inductive UBE: Type → Type :=
  | UB {α: Type} [Inhabited α]: Option String → UBE α

@[simp_itree]
def UBE.handle {E}: UBE ~> OptionT (Fitree E) := fun _ e =>
  match e with
  | UB (some str) => do panic! str; Fitree.Ret none
  | UB none => Fitree.ret none

@[simp_itree]
def UBE.handle! {E}: UBE ~> Fitree E := fun _ e =>
  match e with
  | UB (some str) => panic! str
  | UB none => panic! "Undefined Behavior raised!"

def raiseUB (msg: String) {E α} [Member UBE E] [Inhabited α]: Fitree E α :=
  Fitree.trigger <| UBE.UB (some msg)

def interpUB (t: Fitree UBE R): OptionT (Fitree Void1) R :=
  t.interpOption UBE.handle

def interpUB! (t: Fitree UBE R): Fitree Void1 R :=
  t.interp UBE.handle!

def interpUB' {E} (t: Fitree (UBE +' E) R): OptionT (Fitree E) R :=
  Fitree.interpOption (Fitree.case UBE.handle Fitree.liftHandler) t

def interpUB'! {E} (t: Fitree (UBE +' E) R): Fitree E R :=
  Fitree.interp (Fitree.case UBE.handle! (fun T => @Fitree.trigger E E T _)) t

@[simp] theorem interpUB'_Vis_right:
  interpUB' (Fitree.Vis (Sum.inr e) k) =
  Fitree.Vis e (fun x => interpUB' (k x)) := rfl

@[simp] theorem interpUB'_ret:
  @interpUB' _ E (Fitree.ret r) = Fitree.ret (some r) := rfl

theorem interpUB'_bind (k: T → Fitree (UBE +' E) R):
  interpUB' (Fitree.bind t k) =
  Fitree.bind (interpUB' t) fun x? =>
    match x? with
    | some x => interpUB' (k x)
    | none => Fitree.ret none := by
  apply Fitree.interpOption_bind
