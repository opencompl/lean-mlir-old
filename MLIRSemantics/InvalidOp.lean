/-
## Invalid operation events

This event is a utility event emitted by dialect functions to handle operations
that don't satisfy their invariants. This is not supposed to happen, but the
fallback case allows the functions to be total while keeping the verification
requirement explicit.

This event types comes with a facility interpreter that erases the events, but
only when proven that there are none. The goal is to automate this proof so
that these events go away automatically when programs are well-formed, but
still prevent malformed programs from being studied by accident.
-/

import MLIRSemantics.Fitree

inductive InvalidOpE: Type → Type :=
  | InvalidOp: InvalidOpE Unit

@[simp_itree]
def InvalidOpE.handle {E}: InvalidOpE ~> Fitree E :=
  fun _ ⟨⟩ => Fitree.Ret ()

-- We interpret (InvalidOpE +' E ~> E)

def interp_invalid {E} (t: Fitree (InvalidOpE+'E) R) (H: Fitree.no_event_l t):
    Fitree E R :=
  interp (case_ InvalidOpE.handle (fun T => @Fitree.trigger E E T _)) t
