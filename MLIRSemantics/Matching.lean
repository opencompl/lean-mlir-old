/-
## Pattern matching against MLIR programs

This file implements support for a basic matching system. This system is used
in the framework to concisely express syntactic constraints on operations that
can be resolved by unification. While this cannot express anywhere near all the
constraints, it simplifies the most common ones a great deal.

TODO: Provide the matcher
-/

import MLIRSemantics.Types
import MLIR.AST
open MLIR.AST

/-
### Match term syntax

We have different sorts (types, attributes, values and operations) which are
separated for type safety (specifically variables). However, for simplicity
when dealing with names and reduced confusion, variables names are restricted
to be unique across sorts, ie. we don't allow a value variable and type
variable with the same name.

Unlike usual first-order matching and unification, we don't have any recursive
structure, and instead mostly rely on having independent equations to match
complex patterns.

We assign *substitution priority levels* to variables in the form of natural
numbers. Lower values indicate variables that have been introduced by the user,
while higher values are assigned to automatically-generated variables. When a
substitution occurs, we always substitute variables with higher priority so
that user-assigned names are preserved.
-/

inductive MValue :=
  -- A value variable
  | ValueVar (priority: Nat := 0) (name: String)
  -- A constant value
  | ValueConst (τ: MLIRTy) (const: τ.eval)

inductive MType :=
  -- A type variable
  | TypeVar (priority: Nat := 0) (name: String)
  -- A constant type
  | TypeConst (τ: MLIRTy)

inductive MOp :=
  -- An operation with a known mnemonic. The first list of pairs is arguments
  -- with their types, the second is for the return values.
  | OpKnown (mnemonic: String)
            (args: List (MValue × MType))
            (rets: List (MValue × MType))

-- Common instances

deriving instance Inhabited for MValue
deriving instance Inhabited for MType
deriving instance Inhabited for MOp

def MValue.str: MValue → String
  -- Don't use %STRING to avoid confusion with SSAValue
  | ValueVar _ name => s!"${name}"
  -- TODO: Support printing any value of type τ.eval
  | ValueConst τ v  => s!"(_:{τ})"

instance: ToString MValue := ⟨MValue.str⟩

def MType.str: MType → String
  | TypeVar _ name  => s!"!{name}"
  | TypeConst τ     => s!"{τ}"

instance: ToString MType := ⟨MType.str⟩

def MOp.str: MOp → String
  | OpKnown mnemonic args rets =>
      let str_pair := fun (v, t) => s!"{v}:{t}"
      let str_list l := ", ".intercalate (l.map str_pair)
      s!"{str_list rets} = \"{mnemonic}\"({str_list args})"

instance: ToString MOp := ⟨MOp.str⟩

-- Collect value variables and type variables in a term

def MValue.valueVars: MValue → List String
  | ValueVar _ name => [name]
  | ValueConst _ _  => []

def MType.typeVars: MType → List String
  | TypeVar _ name  => [name]
  | TypeConst _     => []

def MOp.valueVars: MOp → List String
  | OpKnown mnemonic args rets => ((args ++ rets).map (·.fst.valueVars)).join

def MOp.typeVars: MOp → List String
  | OpKnown mnemonic args rets => ((args ++ rets).map (·.snd.typeVars)).join

-- Check whether a variable occurs in a term. We don't need typing here since
-- we have a common pool of unique variable names.

def MValue.occurs (name: String): MValue → Bool
  | ValueVar _ name'  => name' = name
  | _                 => false

def MType.occurs (name: String): MType → Bool
  | TypeVar _ name'   => name' = name
  | _                 => false

def MOp.occurs (name: String): MOp → Bool
  | OpKnown mnemonic args rets =>
      let occurs_pair: MValue × MType → Bool := fun (value, type) =>
        value.occurs name || type.occurs name
      args.any occurs_pair || rets.any occurs_pair

-- Substitute a variable in a term

def MValue.substValue (v: MValue) (name: String) (repl: MValue): MValue :=
  match v with
  | ValueVar _ name'  => if name' = name then repl else v
  | _                 => v

def MType.substType (t: MType) (name: String) (repl: MType): MType :=
  match t with
  | TypeVar _ name'   => if name' = name then repl else t
  | _                 => t

def MOp.substValue (op: MOp) (name: String) (repl: MValue): MOp :=
  match op with
  | OpKnown mnemonic args rets =>
      let args' := args.map (fun (v,t) => (v.substValue name repl, t))
      let rets' := rets.map (fun (v,t) => (v.substValue name repl, t))
      OpKnown mnemonic args' rets'

def MOp.substType (op: MOp) (name: String) (repl: MType): MOp :=
  match op with
  | OpKnown mnemonic args rets =>
      let args' := args.map (fun (v,t) => (v, t.substType name repl))
      let rets' := rets.map (fun (v,t) => (v, t.substType name repl))
      OpKnown mnemonic args' rets'

def MOp.substValues (op: MOp) (repl: List (String × MValue)): MOp :=
  repl.foldl (fun op (name, repl) => op.substValue name repl) op

def MOp.substTypes (op: MOp) (repl: List (String × MType)): MOp :=
  repl.foldl (fun op (name, repl) => op.substType name repl) op
