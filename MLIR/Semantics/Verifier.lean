/- MLIR code verifier

   The verifier checks that language and dialect invariants are verified when
   loading code. It usually provides answers in the form of richer dependent
   types that follow the structure of valid code. -/

import MLIR.AST
open MLIR.AST

/- === Generic verification without changing types ===

   One of the simpler ways to carry the result of verification, without
   changing the type. The verifier is a function [Op → Option α], and the
   vertification certificate simply states that the option is not none. The
   data can be retrieved later.

   The verifier is used like this:

     if H: Verified.ok vFunction op
     then ... -- pass H around
     else ... -- reject operation

   Then a function that has H as parameter can run [Verifier.get H] to access
   the data. -/

def Verifier {α σ ε} (δ: Dialect α σ ε) (R: Type u): Type _ := Op δ → Option R

def Verifier.ok {R} (v: Verifier δ R) (o: Op δ): Bool := Option.isSome (v o)

def Verifier.get {R} {v: Verifier δ R} {o: Op δ}: Verifier.ok v o → R :=
  λ (H: Option.isSome (v o)) =>
    match v o, H with
    | some val, _ => val
    | none, H => nomatch H

-- This is used for the [if H: Verifier.ok vFunction op] syntax
instance {α} (o: Option α): Decidable (Option.isSome o) :=
  match o with
  | some value => isTrue rfl
  | none => isFalse (λ H => by simp [Option.isSome] at H)
