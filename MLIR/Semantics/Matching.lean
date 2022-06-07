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
  -- A dimension (in a vector/tensor). Matches against [Dimension]
  | Dimension
  -- A signedness specification (in integers). Matches against [Signedness]
  | Signedness
  -- A homogeneous list of objects
  | List (s: MSort)

inductive MTerm :=
  -- A typed variable
  | Var (priority: Nat := 0) (name: String) (s: MSort)
  -- A constructor (taken from a fixed pool)
  | App (ctor: String) (args: List MTerm)

-- Common instances

deriving instance Inhabited for MTerm

def MSort.str: MSort → String
  | .Op         => "Op"
  | .MLIRType   => "MLIRType"
  | .SSAVal     => "SSAVal"
  | .AttrValue  => "AttrValue"
  | .Nat        => "Nat"
  | .Dimension  => "Dimension"
  | .Signedness => "Signedness"
  | .List s     => "[" ++ str s ++ "]"

mutual
def MTerm.str: MTerm → String
  | .Var _ name s => "name:" ++ s.str
  | .App ctor args => ctor ++ " " ++ MTerm.strList args

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
  | .App ctor [] => []
  | .App ctor (arg::args) => vars arg ++ vars (.App ctor args)

-- Check whether a variable occurs in a term. We don't check typing here since
-- we have a common pool of unique variable names.
def MTerm.occurs (name: String): MTerm → Bool
  | .Var _ name' _ => name' = name
  | .App ctor [] => false
  | .App ctor (arg::args) => occurs name arg || occurs name (.App ctor args)

-- Substitute a variable in a term
mutual
def MTerm.subst (t: MTerm) (name: String) (repl: MTerm): MTerm :=
  match t with
  | .Var _ name' _ => if name' = name then repl else t
  | .App ctor args => .App ctor (MTerm.substList args name repl)

protected def MTerm.substList (l: List MTerm) (name: String) (repl: MTerm) :=
  match l with
  | [] => []
  | t::ts => subst t name repl :: MTerm.substList ts name repl
end
