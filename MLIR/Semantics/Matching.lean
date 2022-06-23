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
  | MOp
  -- An operation parameter. Matches against (SSAVal × MLIRType δ)
  | MOperand
  -- An MLIR type. Matches against [MLIRType δ]
  | MMLIRType
  -- A value. Matches against [SSAVal]
  | MSSAVal
  -- An attribute. Matches against [AttrVal δ]
  | MAttrValue
  -- A natural number (typically int/float bit size). Matches against [Nat]
  | MNat
  -- A string (in operation names). Matches against [String]
  | MString
  -- A dimension (in a vector/tensor). Matches against [Dimension]
  | MDimension
  -- A signedness specification (in integers). Matches against [Signedness]
  | MSignedness
  -- A homogeneous list of objects
  | MList (s: MSort)

inductive MCtor: List MSort → MSort → Type :=
  -- Integer type
  | INT: MCtor [.MSignedness, .MNat] .MMLIRType
  -- Tensor type
  | TENSOR: MCtor [.MList .MDimension, .MMLIRType] .MMLIRType
  -- Operation with known or unknown mnemonic (TODO: MCtor.OP: unfinished)
  | OP: MCtor [.MString, .MList .MOperand, .MList .MOperand] .MOp
  -- Operation argument of return value
  | OPERAND: MCtor [.MSSAVal, .MMLIRType] .MOperand

  -- SPECIAL CASE: We treat LIST specially in inferSort, to allow variadic
  -- arguments without specifying it here
  | LIST (s: MSort): MCtor [] (.MList s)

inductive MTerm :=
  -- A typed variable
  | Var (priority: Nat := 0) (name: String) (s: MSort)
  -- A constructor. We allow building mistyped terms (but check them later)
  | App {args_sort: List MSort} {ctor_sort: MSort}
        (ctor: MCtor args_sort ctor_sort) (args: List MTerm)
  -- Constants
  | ConstMLIRType (τ: MLIRType builtin)
  | ConstNat (n: Nat)
  | ConstString (s: String)
  | ConstDimension (d: Dimension)
  | ConstSignedness (sgn: Signedness)

-- Accessors

def MCtor.name {s₁ s₂}: MCtor s₁ s₂ → String
  | LIST _  => "LIST"
  | INT     => "INT"
  | TENSOR  => "TENSOR"
  | OP      => "OP"
  | OPERAND => "OPERAND"

-- Common instances

deriving instance Inhabited for MSort
deriving instance Inhabited for MTerm

deriving instance DecidableEq for MCtor
deriving instance DecidableEq for MSort

def MCtor.eq {args_sort₁ ctor_sort₁ args_sort₂ ctor_sort₂}:
    MCtor args_sort₁ ctor_sort₁ → MCtor args_sort₂ ctor_sort₂ → Bool :=
  fun c₁ c₂ =>
    if H: args_sort₁ = args_sort₂ ∧ ctor_sort₁ = ctor_sort₂ then
      cast (by rw [H.1, H.2]) c₁ = c₂
    else
      false

mutual
def MTerm.eq (t₁ t₂: MTerm): Bool :=
  match t₁, t₂ with
  | Var _ name₁ s₁, Var _ name₂ s₂ =>
      name₁ = name₂ && s₁ = s₂
  | App ctor₁ args₁, App ctor₂ args₂ =>
      MCtor.eq ctor₁ ctor₂ && eqList args₁ args₂
  | _, _ => false

def MTerm.eqList (l₁ l₂: List MTerm): Bool :=
  match l₁, l₂ with
  | [], [] => true
  | t₁::l₁, t₂::l₂ => eq t₁ t₂ && eqList l₁ l₂
  | _, _ => false
end

instance: BEq MTerm where
  beq := MTerm.eq

def MSort.str: MSort → String
  | .MOp         => "Op"
  | .MOperand    => "Operand"
  | .MMLIRType   => "MLIRType"
  | .MSSAVal     => "SSAVal"
  | .MAttrValue  => "AttrValue"
  | .MNat        => "Nat"
  | .MString     => "String"
  | .MDimension  => "Dimension"
  | .MSignedness => "Signedness"
  | .MList s     => "[" ++ s.str ++ "]"

mutual
def MTerm.str: MTerm → String
  -- Short notations for common sorts of variables
  | .Var _ name .MMLIRType => "!" ++ name
  | .Var _ name .MSSAVal => "%" ++ name
  -- General notation
  | .Var _ name s => "name:" ++ s.str
  | .App ctor args => ctor.name ++ " [" ++ MTerm.strList args ++ "]"
  -- Constants
  | ConstMLIRType c
  | ConstNat c
  | ConstString c
  | ConstDimension c
  | ConstSignedness c =>
      toString c

protected def MTerm.strList: List MTerm → String
  | [] => ""
  | [t] => str t
  | t::ts => str t ++ ", " ++ MTerm.strList ts
end

instance: ToString MSort where
  toString := MSort.str
instance: ToString MTerm where
  toString := MTerm.str

-- Collect variables in a term
def MTerm.vars: MTerm → List String
  | .Var _ name _ => [name]
  | .App ctor [] => []
  | .App ctor (arg::args) =>
      vars arg ++ vars (.App ctor args)
  | _ => []

-- Collect variables and their sorts
def MTerm.varsWithSorts: MTerm → List (String × MSort)
  | .Var _ name sort => [(name, sort)]
  | .App ctor [] => []
  | .App ctor (arg::args) =>
      varsWithSorts arg ++ varsWithSorts (.App ctor args)
  | _ => []

-- Check whether a variable occurs in a term. We don't check typing here since
-- we have a common pool of unique variable names.
def MTerm.occurs (name: String): MTerm → Bool
  | .Var _ name' _ => name' = name
  | .App ctor [] => false
  | .App ctor (arg::args) =>
      occurs name arg || occurs name (.App ctor args)
  | _ => false

-- Substitute a variable in a term
mutual
def MTerm.subst (t: MTerm) (name: String) (repl: MTerm): MTerm :=
  match t with
  | .Var _ name' _ => if name' = name then repl else t
  | .App ctor args => .App ctor (MTerm.substList args name repl)
  | t => t

protected def MTerm.substList (l: List MTerm) (name: String) (repl: MTerm) :=
  match l with
  | [] => []
  | t::ts => subst t name repl :: MTerm.substList ts name repl
end

-- Substitue a set of variables in a term
def MTerm.substVars (t: MTerm) (repl: List (String × MTerm)): MTerm :=
  repl.foldl (fun t (name, repl) => t.subst name repl) t

/-
### Sort inference

In order to ensure we only manipulate well typed match terms and equalities
despite mixing constructors, we aggressively check typing during matching and
unification.
-/

mutual
def MTerm.inferSort: MTerm → Option MSort
  | Var _ _ s => some s
  | App (.LIST s) args => do
      let l ← inferSortList args
      if l.all (· = s) then some (.MList s) else none
  | @App args_sort ctor_sort ctor args =>
      if args.length != args_sort.length then
        none
      else if inferSortList args |>.isEqSome args_sort then
        some ctor_sort
      else
        none
  | ConstMLIRType _     => some .MMLIRType
  | ConstNat _          => some .MNat
  | ConstString _       => some .MString
  | ConstDimension _    => some .MDimension
  | ConstSignedness _   => some .MSignedness

def MTerm.inferSortList: List MTerm → Option (List MSort)
  | [] => some []
  | t::l => do return (← inferSort t) :: (← inferSortList l)
end
