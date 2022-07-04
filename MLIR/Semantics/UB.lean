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
import MLIR.Dialects
import MLIR.AST
open MLIR.AST


@[extern c "lean_panic_fn"]
private def voidPanic (msg : String) : Void := sorry

inductive UBE: Type → Type :=
  | UB: UBE Void
  | DebugUB: String → UBE Void


@[simp_itree]
def UBE.handle {E}: UBE ~> OptionT (Fitree E) := fun _ e =>
  match e with
  | UB => Fitree.Ret none
  | DebugUB str => do panic! str; Fitree.Ret none

@[simp_itree]
def UBE.handle! {E}: UBE ~> Fitree E := fun _ e => do
  match e with
  | UB => return (voidPanic "Undefined Behavior raised!")
  | DebugUB str => return (voidPanic str)

-- We interpret (UBE +' E ~> E)

@[simp_itree]
private def optionT_defaultHandler: E ~> OptionT (Fitree E) :=
  fun _ e => OptionT.lift $ Fitree.trigger e

def interp_ub {E} (t: Fitree (UBE +' E) R): OptionT (Fitree E) R :=
  interp (Fitree.case_ UBE.handle optionT_defaultHandler) t

def interp_ub! {E} (t: Fitree (UBE +' E) R): Fitree E R :=
  interp (Fitree.case_ UBE.handle! (fun T => @Fitree.trigger E E T _)) t
