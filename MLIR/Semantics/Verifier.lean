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

universe u

def Verifier (α: Type u): Type _ := Op → Option α

def Verifier.ok {α} (v: Verifier α) (o: Op) := Option.isSome (v o)

def Verifier.get {α} {v: Verifier α} {o: Op}: Verifier.ok v o → α :=
  λ (H: Option.isSome (v o)) =>
    match v o, H with
    | some val, _ => val
    | none, H => nomatch H

-- This is used for the [if H: Verifier.ok vFunction op] syntax
instance {α} (o: Option α): Decidable (Option.isSome o) :=
  match o with
  | some value => isTrue rfl
  | none => isFalse (λ H => by simp [Option.isSome] at H)


/- === Consistency between arity by argument count and operation type ===

   Assuming custom syntax is expanded, the number of arguments and the
   functional type of the operation must match. Equality can be structurally
   enforced by weaving the lists together, morally changing

     "op"(%arg, %arg, %arg): (!type, !type, !type) -> !ret

   into the woven form

     "op"(%arg:!type, %arg:!type, %arg!type): !ret

   The number of arguments can also be specified at the same time. -/

private def zip_args_types (args: List SSAVal) (ty: MLIRTy) :=
  match args, ty with
  | [], MLIRTy.tuple [] =>
      some []
  | a::args, MLIRTy.tuple (t::tys) =>
      Option.map ((a,t) :: .) (zip_args_types args (MLIRTy.tuple tys))
  | [a], t =>
      some [(a,t)]
  | _, _ =>
      none

-- This is the verifier function
def vArgTypeArity: Verifier (List (SSAVal × MLIRTy)) :=
  λ op =>
    match op with
    | Op.mk _ args _ _ _ (MLIRTy.fn ty _) => zip_args_types args ty
    | _ => none
