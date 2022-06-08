/-
## Pattern matching against MLIR programs

This file implements support for a basic matching system. This system is used
in the framework to concisely express syntactic constraints on operations that
can be resolved by unification. While this cannot express anywhere near all the
constraints, it simplifies the most common ones a great deal.

TODO: Provide the matcher
-/

import MLIR.Semantics.Types
import MLIR.Dialects.BuiltinModel
import MLIR.AST
open MLIR.AST

/-
### Match term syntax

We have different sorts (operations, values, types, attributes, etc) with typed
variables. This allows us to know types everywhere in terms, and thus avoid
typing mishaps due to bugs (which would be hard to debug), even if we only have
one type for all terms.

Unlike usual first-order matching and unification, we don't have a deeply
recursive structure, and instead mostly rely on having independent equations to
match complex patterns.

We assign *substitution priority levels* to variables in the form of natural
numbers. Lower values indicate variables that have been introduced by the user,
while higher values are assigned to automatically-generated variables. When a
substitution occurs, we always substitute variables with higher priority so
that user-assigned names are preserved.
-/

inductive MSort :=
  -- An MLIR operation. Matches against [Op δ]
  | Op
  -- An MLIR type. Matches against [MLIRType δ]
  | MLIRType
  -- A value. Matches against [SSAVal]
  | SSAVal
  -- An attribute. Matches against [AttrVal δ]
  | AttrValue
  -- A natural number (typically int/float bit size). Matches against [Nat]
  | Nat
  -- A string (in operation names). Matches against [String]
  | String
  -- A dimension (in a vector/tensor). Matches against [Dimension]
  | Dimension
  -- A signedness specification (in integers). Matches against [Signedness]
  | Signedness
  -- A homogeneous list of objects
  | List (s: MSort)

inductive MCtor: List MSort → MSort → Type :=
  | INT: MCtor [.Signedness, .Nat] .MLIRType
  | TENSOR: MCtor [.List .Dimension, .MLIRType] .MLIRType
  -- TODO: Incomplete op
  | OP: MCtor [.String, .List .SSAVal, .List .MLIRType] .Op
  -- TODO: Varargs for LIST? >_o
  -- | LIST (s: MSort): MCtor [.List s]

inductive MTerm :=
  -- A typed variable
  | Var (priority: Nat := 0) (name: String) (s: MSort)
  -- A constructor. We allow building mistyped terms (but check them later)
  | App (args_sort: List MSort) (ctor_sort: MSort)
        (ctor: MCtor args_sort ctor_sort) (args: List MTerm)

-- Accessors

def MCtor.name {s₁ s₂}: MCtor s₁ s₂ → String
  | INT => "INT"
  | TENSOR => "TENSOR"
  | OP => "OP"

-- Common instances

deriving instance Inhabited for MSort
deriving instance Inhabited for MTerm

deriving instance DecidableEq for MCtor
deriving instance DecidableEq for MSort

mutual
def MTerm.eq (t₁ t₂: MTerm): Bool :=
  match t₁, t₂ with
  | Var _ name₁ s₁, Var _ name₂ s₂ =>
      name₁ = name₂ && s₁ = s₂
  | App args_sort₁ ctor_sort₁ ctor₁ args₁,
    App args_sort₂ ctor_sort₂ ctor₂ args₂ =>
      if H: args_sort₁ = args_sort₂ ∧ ctor_sort₁ = ctor_sort₂ then
        cast (by rw [H.1, H.2]) ctor₁ = ctor₂ && eqList args₁ args₂
      else
        false
  | _, _ => false

def MTerm.eqList (l₁ l₂: List MTerm): Bool :=
  match l₁, l₂ with
  | [], [] => true
  | t₁::l₁, t₂::l₂ => eq t₁ t₂ && eqList l₁ l₂
  | _, _ => false
end

instance: BEq MTerm where
  beq := MTerm.eq

def MSort_str: MSort → String
  | .Op         => "Op"
  | .MLIRType   => "MLIRType"
  | .SSAVal     => "SSAVal"
  | .AttrValue  => "AttrValue"
  | .Nat        => "Nat"
  | .String     => "String"
  | .Dimension  => "Dimension"
  | .Signedness => "Signedness"
  | .List s     => "[" ++ MSort_str s ++ "]"

def MSort.str := MSort_str

mutual
def MTerm.str: MTerm → String
  | .Var _ name s => "name:" ++ s.str
  | .App _ _ ctor args => ctor.name ++ " " ++ MTerm.strList args

protected def MTerm.strList: List MTerm → String
  | [] => ""
  | t::ts => str t ++ " " ++ MTerm.strList ts
end

instance: ToString MSort where
  toString := MSort.str
instance: ToString MTerm where
  toString := MTerm.str

-- Collect variables in a term

def MTerm.vars: MTerm → List String
  | .Var _ name _ => [name]
  | .App _ _ ctor [] => []
  | .App _ _ ctor (arg::args) =>
      vars arg ++ vars (.App _ _ ctor args)

-- Check whether a variable occurs in a term. We don't check typing here since
-- we have a common pool of unique variable names.
def MTerm.occurs (name: String): MTerm → Bool
  | .Var _ name' _ => name' = name
  | .App _ _ ctor [] => false
  | .App _ _ ctor (arg::args) =>
      occurs name arg || occurs name (.App _ _ ctor args)

-- Substitute a variable in a term
mutual
def MTerm.subst (t: MTerm) (name: String) (repl: MTerm): MTerm :=
  match t with
  | .Var _ name' _ => if name' = name then repl else t
  | .App _ _ ctor args => .App _ _ ctor (MTerm.substList args name repl)

protected def MTerm.substList (l: List MTerm) (name: String) (repl: MTerm) :=
  match l with
  | [] => []
  | t::ts => subst t name repl :: MTerm.substList ts name repl
end

/-
### Sort inference

In order to ensure we only manipulate well typed match terms and equalities
despite mixing constructors, we aggressively check typing during matching and
unification.
-/

mutual
def MTerm.inferSort: MTerm → Option MSort
  | Var _ _ s => some s
  | App args_sort ctor_sort ctor args =>
      if args.length != args_sort.length then
        none
      else if inferSortList args |>.isEqSome args_sort then
        some ctor_sort
      else
        none

def MTerm.inferSortList: List MTerm → Option (List MSort)
  | [] => some []
  | t::l => do return (← inferSort t) :: (← inferSortList l)
end
