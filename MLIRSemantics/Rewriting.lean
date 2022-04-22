/-
## Rewriting MLIR fragments with PDL

This file implements some framework support for PDL rewrites, which is used to
formalize and prove rewrite operations.

This formalization works by treating a PDL rewrite as a first-order unification
problem. In order to prove a rewrite, we first consider a most general solution
to the unification problem, and then add any other constraints as hypotheses of
the correction theorem.

The general idea is to take as input a PDL operation, and output a template
theorem that looks like this:

```lean4
-- (unification variables)
∀ (value_1: Value) (type_1: Type) (op_1: Operation),
  -- (non-unification-based constraints)
  is_float_type type_1 →

  semantics_of [mlir_bb|
    -- (symbolic pattern, almost verbatim from the `pdl.pattern` description)
    op_1 (value_1: type_1) 2
  ] =
  semantics_of [mlir_bb|
    -- (rewritten version of above)
    output_op_1 ...
  ]
```

For readability/accessibility reasons, this statement would probably be
synthetized into Lean code rather than being a general theorem instantiated
with that particular PDL rewrite. The second option would require considering
"any matching basic block" and then painstakingly proving that such a match has
the structure underlined by the PDL rewrite, which seems annoying.

Operations of PDL that we should partially or totally express:

* `pdl.attribute`:
  Used as unification variable. TODO: should support type/value requirements
* `pdl.erase`:
  One of the rewriting actions
* `pdl.operand`:
  Used as unification variable. TODO: should support type requirement
* `pdl.operation`:
  Used as unification variable and as rewriting output.
  TODO: should support type/value/attributes, for fairly obvious reasons.
  Ranges are not important for now.
* `pdl.pattern`:
  Input operation for the whole rewriting workflow.
* `pdl.replace`:
  One of the rewriting actions. TODO: should support both values and operations
* `pdl.result`:
  Utility to reference results from `pdl.operation` pointers
* `pdl.rewrite`:
  One of the main components of the specification. We might not care about the
  root selection since we don't implement the pattern matching/search.
* `pdl.type`:
  Used as unification variable.

Operations or features of PDL that we don't try to express (yet):

* `pdl.operands`, `pdl.results`, `pdl.types`
  Anything related to ranges
* `pdl.apply_native_constraint`, `pdl.apply_native_rewrite`:
  This is possible in principle, but requires the native constraints/rewrites
  to be rewritten in Lean, which is slightly inconvenient.
* Use of native rewrite function in `pdl.rewrite`:
  Same as above
* Using variadic arguments or matching operands with variadic arguments
-/

import MLIRSemantics.Types
import MLIR.AST
import MLIR.EDSL

open MLIR.AST
open MLIR.EDSL


/-
### Syntax of the unification problem

For this problem, we have different sorts: type, attributes, values (operands)
and operations. We could handle them with syntactically different terms, but
each new type of term and each new type of variable would add new occurence
check functions, substitution functions, fresh name tests, etc. So instead we
use an untyped term structure (the sort of each variable can be determined by
context anyway).

Unlike usual first-order unification, we don't have any recursive structure,
and instead mostly rely on independent equations.

There is quite a lot of slack in the exact shape of the equations, leading to
different versions of the problem. This is only one of the options.
-/

inductive UTerm :=
  -- A variable (either value or type, depending on context)
  | Var: String → UTerm
  -- A constant type
  | ConstType: MLIRTy → UTerm
  -- A constant value
  | ConstVal: (τ: MLIRTy) → τ.eval → UTerm
  -- An operation with a known mnemonic. The first list of pairs is arguments
  -- with their types, the second is for the return values
  | KnownOp: String → List (UTerm × UTerm) → List (UTerm × UTerm) → UTerm
deriving Inhabited

def UEq := UTerm × UTerm

mutual
  def UTerm.free_vars: UTerm → List String
    | Var s                 => [s]
    | ConstType _           => []
    | ConstVal _ _          => []
    | KnownOp _ args rets   => free_vars_list args ++ free_vars_list rets
  private def UTerm.free_vars_list: List (UTerm × UTerm) → List String
    | []                    => []
    | (v, t) :: l           => free_vars v ++ free_vars t ++ free_vars_list l 
end

mutual
  def UTerm.occurs (name: String): UTerm → Bool
    | Var s                 => s = name
    | ConstType _           => false
    | ConstVal _ _          => false
    | KnownOp op args rets  => occurs_list name args || occurs_list name rets
  private def UTerm.occurs_list (name: String): List (UTerm × UTerm) → Bool
    | []                    => false
    | (v, t) :: l           => occurs name v || occurs name t ||
                               occurs_list name l
end

def UEq.occurs: UEq → String → Bool
  | (t₁, t₂), name => t₁.occurs name || t₂.occurs name

mutual
  def UTerm.subst (t: UTerm) (name: String) (t': UTerm): UTerm :=
    match t with
    | Var s                 => if s = name then t' else Var s
    | ConstType _           => t
    | ConstVal _ _          => t
    | KnownOp op args rets  => KnownOp op (subst_list args name t')
                                          (subst_list rets name t')
  private def UTerm.subst_list (l: List (UTerm × UTerm)) name t' :=
    match l with
    | []                    => []
    | (v, t) :: l           => (subst v name t', subst t name t') ::
                               subst_list l name t'
end

def UTerm.subst_all (t: UTerm) (repl: List (String × UTerm)) :=
  repl.foldl (fun t (name, t') => t.subst name t') t

def UEq.subst (e: UEq) (name: String) (t': UTerm): UEq :=
  match e with
  | (t₁, t₂) => (t₁.subst name t', t₂.subst name t')

structure Unification where mk ::
  -- Current set of equations
  eqns: List UEq
  -- Substitutions performed so far
  substs: List UEq
  -- All variables defined so far (for fresh name generation)
  allvars: List String
  -- Typing judgements (to be inserted into operation terms)
  judgements: List (String × UTerm)
  -- Names of results for each operation, and their types
  opresults: List (String × List (String × UTerm))
deriving Inhabited

def Unification.empty: Unification :=
  { eqns        := [],
    substs      := [],
    allvars     := [],
    judgements  := [],
    opresults   := [], }

def Unification.find_judgement (u: Unification) (s: String): Option UTerm :=
  u.judgements.findSome? (fun (v, t) => if v = s then some t else none)

def Unification.find_judgements (u: Unification) (l: List String):
    Option (List (String × UTerm)) :=
  match l with
  | [] => some []
  | s :: l =>
    (u.find_judgement s).bind fun t =>
    (find_judgements u l).bind fun l =>
    some $ (s, t) :: l

def Unification.find_opresult (u: Unification) (op: String) (idx: Nat) :=
  u.opresults.findSome? (fun (n, l) => if n = op then l.get? idx else none)


/- String representations -/

mutual
  def UTerm.str: UTerm → String
    | UTerm.Var s =>
        s!"%{s}"
    | UTerm.ConstType τ =>
        toString τ
    | UTerm.ConstVal τ v =>
        s!"(TODO:{τ})"
    | UTerm.KnownOp name vals rets =>
        s!"{str_list rets} = \"{name}\"({str_list vals})"

  private def UTerm.str_list: List (UTerm × UTerm) → String
    | [] => ""
    | [(v,t)] => str v ++ ":" ++ str t
    | (v,t) :: l => str v ++ ":" ++ str t ++ ", " ++ str_list l
end

instance: ToString UTerm := ⟨UTerm.str⟩

def UEq.str: UEq → String
  | (t₁, t₂) => s!"{t₁} ≡ {t₂}"

instance: ToString UEq := ⟨UEq.str⟩

def Unification.str: Unification → String := fun u =>
  "\n".intercalate (u.eqns.map toString)

instance: ToString Unification := ⟨Unification.str⟩

def Unification.str_full: Unification → String := fun u =>
  "Equations:\n" ++ (String.join $ u.eqns.map (s!"  {·}\n")) ++
  "Substitutions:\n" ++ (String.join $ u.substs.map (s!"  {·}\n")) ++
  "All variables:\n " ++ (String.join $ u.allvars.map (s!" %{·}")) ++
  "\nTyping judgements:\n" ++ (String.join $ u.judgements.map
    (fun (v,t) => s!"  %{v}: {t}\n")) ++
  "Operation results:\n" ++ (String.join $ u.opresults.map
    (fun (n,l) => s!"  %{n}: {l}\n"))


/-
### Unification algorithm

We use a naive unification algorithm. Given the size of the rewrites at play,
we don't really care about performance, and instead prefer a more natural
traversal of the structure that leads to more intuitive logs and error
messages.

This is algorithm 1 of [1], which essentially normalizes the set of equations
through a number of unifier-preserving transformations.

TODO: Have "priorities" on variables so that automatically-named variables are
      substituted first and user-name variables are kept instead!

[1] Martelli, Alberto, and Ugo Montanari. "An efficient unification algorithm."
    ACM Transactions on Programming Languages and Systems (TOPLAS) 4.2 (1982):
    258-282.
-/

open UTerm

-- Transformation (a): turn [t = x] (t not a variable) into [x = t]

private def orient_one (eqn: UEq): IO (UEq × Bool) :=
  match eqn with
  | (Var v₁, Var v₂) =>
      return (eqn, false) -- TODO: Variable priority
  | (t₁, Var v₂) => do
      IO.print s!"Orient: {eqn}\n\n"
      return ((Var v₂, t₁), true)
  |_ =>
      return (eqn, false)

private def orient (eqns: List UEq): IO (List UEq × Bool) :=
  eqns.foldlM
    (fun (trs, b) eqn => do
      let (eqn, b') ← orient_one eqn
      return (trs ++ [eqn], b || b'))
    ([], false)

-- Transformation (b): erase [x = x] (x a variable)

private def erase_filter: UEq → Bool
  | (Var v₁, Var v₂) =>
      v₁ = v₂
  | _ =>
      false

private def erase (eqns: List UEq): List UEq × Bool :=
  let eqns' := eqns.filter (fun eq => ! erase_filter eq)
  (eqns', eqns'.length != eqns.length)

-- Transformation (c): reduce [t = t'], where t and t' are constructed, to
-- equality of arguments (or no solution if the constructors differ)
-- TODO: Also break up interesting type and value equalities

private def reduce_one (eqn: UEq): IO (Option (List UEq × Bool)) :=
  match eqn with
  | (KnownOp name₁ vals₁ rets₁, KnownOp name₂ vals₂ rets₂) => do
      if name₁ = name₂
          ∧ vals₁.length = vals₂.length
          ∧ rets₁.length = rets₂.length then
        IO.print s!"Reduce: {eqn}\n\n"
        return some (List.join (
            List.map₂ (fun (v₁,t₁) (v₂,t₂) => [(v₁,v₂),(t₁,t₂)]) vals₁ vals₂ ++
            List.map₂ (fun (v₁,t₁) (v₂,t₂) => [(v₁,v₂),(t₁,t₂)]) rets₁ rets₂),
          true)
      else
        return none
  | eq =>
      return some ([eq], false)

private def reduce (eqns: List UEq): IO (Option (List UEq × Bool)) :=
  eqns.foldlM
    (fun acc eqn => do
       match ← reduce_one eqn with
       | some (eqns, b') =>
          return acc.bind fun (trs, b) => some (trs ++ eqns, b || b')
       | none =>
          return none)
    $ some ([], false)

-- Transformation (d): eliminate [x = t] by substituting x if it's used
-- elsewhere and does not occur in t

private def elim_at (eqns: List UEq) (n: Nat):
    IO (Option (List UEq × List UEq)) :=
  if H: n < eqns.length then
    let eqn := eqns.get ⟨n, H⟩
    let others := (eqns.enum.filter (·.1 ≠ n)).map (·.snd)

    match eqn with
    | (Var v₁, t₂) =>
        if t₂.occurs v₁ then do
          IO.println s!"Equation {eqn} has a cycle!"
          return none -- cycle
        else if eqns.enum.any (fun (j,eqn) => j ≠ n ∧ eqn.occurs v₁) then do
          IO.print s!"Substitute: {eqn}\n\n"
          return some (others.map (·.subst v₁ t₂), [eqn])
        else
          return some (eqns, [])
    | _ =>
        return some (eqns, [])
  else
    return some (eqns, [])

private def elim (eqns: List UEq): IO (Option (List UEq × List UEq)) :=
  (List.range eqns.length).foldlM
    (fun acc n => do
      match acc with
      | some (eqns, substs) =>
          match ← elim_at eqns n with
          | some (eqns', substs') =>
              return some (eqns', substs ++ substs')
          | none =>
              return none
      | none =>
          return none)
    $ some (eqns, [])


-- Unification main loop: greedily applies each transformation in order

def Unification.simplify: Unification → IO (Option (Unification × Bool)) :=
  fun u => do
    -- Orient all rules
    let (eqns, b) ← orient u.eqns
    if b then return some ({u with eqns := eqns}, true) else
    -- Substitute all intermediate variables
    match (← elim eqns) with
    | some (eqns, []) =>
        -- Match arguments and return values of common operations
        match (← reduce eqns) with
        | some (eqns, b) => return some ({u with eqns := eqns}, b)
        | none => return none
    | some (eqns, substs) =>
        return some ({ u with eqns := eqns, substs := u.substs ++ substs},true)
    | none =>
        return none

partial def Unification.solve (u: Unification) (n: Nat):
    IO (Option Unification) := do
  IO.print s!"{u}\n\n"
  match ← u.simplify with
  | some (u, b) =>
      -- Clean up after every step
      let u: Unification := {u with eqns := (erase u.eqns).fst}
      if b then
        if n > 0 then
          let u ← u.solve (n-1)
          return u
        else
          IO.println s!"Max iterations reached"
          return u
      return u
  | none =>
      IO.println s!"Problem has no solution!"
      return none

def Unification.apply (solved_u: Unification): UTerm → UTerm := fun t =>
  List.foldl
    (fun t eqn =>
      match eqn with
      | (Var n₁, t₂) => t.subst n₁ t₂
      | _ => t)
    t (solved_u.substs ++ solved_u.eqns)



/-
### Basic example

Here, we consider an under-specified [x*2] pattern (ex_root) that we presumably
want to turn into [x+x]. The pattern doesn't specify that x is an i32 as this
is implicit, and we uncover this fact by unifying with the general shape of a
multiplication operation (mul_pattern), supposedly obtained from IRDL.
-/

private def mul_pattern: UTerm :=
  KnownOp "arith.mul"
    [(Var "op_x", Var "T"), (Var "op_y", Var "T")]
    [(Var "op_res", Var "T")]

-- %two = pdl.value 2: i32
private def ex_two: UEq :=
  (Var "two", ConstVal (MLIRTy.int 32) 2)

-- %x = pdl.value
-- %root = "arith.mul"(%x, %two)
-- (%x is implicit, while %x_T, %_0 and %_0_T are automatically inserted)
private def ex_root: UTerm :=
  KnownOp "arith.mul"
    [(Var "x", Var "x_T"),
     (Var "two", ConstType (MLIRTy.int 32))]
    [(Var "_0", Var "_0_T")]

private def mul_example: Unification :=
  { Unification.empty with eqns := [ex_two, (mul_pattern, ex_root)] }

#eval show IO Unit from do
  let u ← mul_example.solve 999
  let stmt := u.get!.apply ex_root
  IO.println s!"Theorem input:\n{stmt}"


/-
### Conversion of PDL programs to Unification problems

The interpretation of a PDL program as a unification problem is fairly
straightforward; most PDL operations simply add new names or equations to the
set. We still need to find a template to unify operations against.

PDL has a *lot* of restrictions on what you can write; you can't use the same
variable at two different places (implicit unification), you can't have a
declaration operation (pdl.value, pdl.type, etc) without binding it to a
variable, etc. This allows us to make assumptions on the shape of the input.
-/

def MLIR.AST.SSAVal.str: MLIR.AST.SSAVal → String
  | SSAVal.SSAVal s => s

-- Make up to [n] attempts at finding a fresh name by suffixing [s]
def Unification.fresh_name_numbered (u: Unification) (s: String) (n p: Nat) :=
  match n with
  | 0 => s
  | m+1 =>
      let s' := s!"{s}{p}"
      if u.allvars.all (· != s') then
        s'
      else
        fresh_name_numbered u s m (p+1)

-- Generate a new fresh name based on [name]; if not available, resort to
-- adding numbered suffixes
def Unification.fresh_name (u: Unification) (name: String): String :=
  if u.allvars.all (· != name) then
    name
  else
    u.fresh_name_numbered name (u.allvars.length+1) 0

-- Generate [n] fresh names based on [name]
def Unification.fresh_names (u: Unification) (name: String) (n: Nat):
    List String × Unification :=
  (List.range n).foldl
    (fun (names, u) idx =>
      let n' := u.fresh_name (name ++ toString idx)
      (names ++ [n'], { u with allvars := u.allvars ++ [n'] }))
    ([], u)

-- Generate a copy of the term with the specified prefix and fresh names.
-- Returns a pair with the new term and the list of all variables involved.
def Unification.fresh_term (u: Unification) (prefix_: String) (t: UTerm) :=
  let vars := t.free_vars
  -- Remember generated names along the way to not generate them twice
  let (repl, done, u) := vars.foldl
    (fun (repl, done, u) var =>
      if var ∈ done then
        (repl, done, u)
      else
        let var' := u.fresh_name (prefix_ ++ var)
        (repl ++ [(var, Var var')], done ++ [var],
         {u with allvars := u.allvars ++ [var']}))
    ([], [], u)
  (t.subst_all repl, u)

namespace PDL2U

-- Get the n-th category of arguments from the specified variadic list by using
-- the operand segment size attribute
private def operand_segment (args: List SSAVal) (oss: TensorElem) (n: Nat) :=
  -- Flatten segment sizes
  let oss := match oss with
  | TensorElem.nested l => l.map (
      match · with | TensorElem.int i => i.toNat | _ => 0)
  | _ => []
  -- Accumulate
  let (oss, _) := oss.foldl
    (fun (segments, acc) size => (segments ++ [(acc, size)], acc + size))
    ([], 0)
  -- Get the segment for that argument
  let (start, size) := oss.getD n (0,0)
  (List.range size).filterMap (fun i => args.get? (i + start))

private def handle_pdl_stmt (ins_patterns: List UTerm) (u: Unification)
    (stmt: BasicBlockStmt): IO Unification :=
  match stmt with
  | BasicBlockStmt.StmtAssign name op => match op with

    -- %name = pdl.type
    | Op.mk "pdl.type" [] [] [] (AttrDict.mk [])
          (MLIRTy.fn (MLIRTy.tuple [])
                     (MLIRTy.user "pdl.type")) => do
        IO.println s!"Found new type variable: {name}"
        return { u with allvars := u.allvars ++ [name.str] }

    -- %name = pdl.type: TYPE
    | Op.mk "pdl.type" [] [] [] (AttrDict.mk [
            AttrEntry.mk "type" (AttrVal.type τ)
          ])
          (MLIRTy.fn (MLIRTy.tuple [])
                     (MLIRTy.user "pdl.type")) => do
        IO.println s!"Found new type variable: {name} (= {τ})"
        let eqn: UEq := (Var name.str, ConstType τ)
        return { u with eqns := u.eqns ++ [eqn],
                        allvars := u.allvars ++ [name.str] }

    -- %name = pdl.operand
    | Op.mk "pdl.operand" [] [] [] (AttrDict.mk [])
          (MLIRTy.fn (MLIRTy.tuple [])
                     (MLIRTy.user "pdl.value")) => do
        let name_type := u.fresh_name (name.str ++ "_T")
        let j := (name.str, Var name_type)
        IO.println s!"Found new variable: {name} of type {name_type}"
        return { u with judgements := u.judgements ++ [j],
                        allvars := u.allvars ++ [name.str, name_type] }

    -- %name = pdl.operand: %name_type
    | Op.mk "pdl.operand" [name_type] [] [] (AttrDict.mk [])
          (MLIRTy.fn (MLIRTy.tuple [MLIRTy.user "pdl.type"])
                     (MLIRTy.user "pdl.value")) => do
        IO.println s!"Found new variable: {name} of type {name_type}"
        let j := (name.str, Var name_type.str)
        return { u with judgements := u.judgements ++ [j],
                        allvars := u.allvars ++ [name.str] }

    -- %name = pdl.operation "OPNAME"(ARGS) -> TYPE
    | Op.mk "pdl.operation" args [] [] attrs some_type => do
        let (attributeNames, opname, operand_segment_sizes) :=
          (attrs.find "attributeNames",
           attrs.find "name",
           attrs.find "operand_segment_sizes")

        match attributeNames, opname, operand_segment_sizes with
        | some (AttrVal.list attributeNames),
          some (AttrVal.str opname),
          some (AttrVal.dense oss (MLIRTy.vector _ _)) =>
            let values := operand_segment args oss 0
            let types  := operand_segment args oss 2
            IO.println s!"Found new operation: {name} matching {opname}"

            let valuesT := u.find_judgements (values.map (·.str))
            if valuesT.isNone then
              IO.println s!"→ Some of the values ({values}) are undeclared!"
              return u
            let valuesT := valuesT.get!

            let (rets, u) := u.fresh_names (name.str ++ "_res") types.length

            IO.println s!"→ Values: {values}, with types: {valuesT}"
            IO.println s!"→ Return types: {types}, with names: {rets}"

            let ins := ins_patterns.find? (fun t => match t with
              | KnownOp n _ _ => n = opname
              | _ => false)
            match ins with
            | some ins =>
                IO.println s!"→ Using pattern: {ins}"
                let (ins, u) := u.fresh_term s!"{name.str}_" ins
                IO.println s!"→ Specialized pattern: {ins}"

                let args  := valuesT.map (fun (v,t) => (Var v, t))
                let rets1 := List.zip rets (types.map (Var ∘ (·.str)))
                let rets2 := List.zip (rets.map Var) (types.map (Var ∘ (·.str)))
                let eqn: UEq := (KnownOp opname args rets2, ins)
                -- TODO: Lots of variables ignored?
                return { u with eqns := u.eqns ++ [eqn],
                                allvars := u.allvars ++ [name.str],
                                opresults := u.opresults ++ [(name.str,rets1)]}
            | _ =>
                IO.println s!"No pattern known for instruction {opname}!"
                return u
        | _, _, _ =>
            IO.println s!"unexpected attributes: {attrs}"
            return u

    -- %name = pdl.result INDEX of %op
    | Op.mk "pdl.result" [arg] [] [] attrs
          (MLIRTy.fn (MLIRTy.tuple [MLIRTy.user "pdl.operation"])
                     (MLIRTy.user "pdl.value")) => do
        let index := attrs.find "index"
        match index with
        | some (AttrVal.int index (MLIRTy.int _)) =>
            IO.println (s!"Found new variable: {name} aliasing result " ++
              s!"{index} of {arg}")
            let result_info := u.find_opresult arg.str (index.toNat)
            if result_info.isNone then
              IO.println "→ No such result!"
              return u

            let (res_name, res_type) := result_info.get!
            let eqn: UEq := (Var name.str, Var res_name)
            let j := (name.str, res_type)
            return { u with eqns := u.eqns ++ [eqn],
                            allvars := u.allvars ++ [name.str],
                            judgements := u.judgements ++ [j] }
        | _ =>
            IO.println s!"unexpected attributes on {op}: {attrs}"
            return u

    | _ => do
        IO.println s!"unrecognized PDL operation {op}"
        return u

  | BasicBlockStmt.StmtOp op => match op with
    -- TODO: pdl.rewrite ...
    | _ => do
        IO.println s!"unrecognized PDL operation {op}"
        return u

def convert (pattern: Op) (ins_patterns: List UTerm):
    IO (Option Unification) :=
  match pattern with
  | Op.mk "pdl.pattern" [] [] [region] attrs ty =>
      match region with
      | Region.mk [BasicBlock.mk name [] stmts] =>
          return some $ ← stmts.foldlM (handle_pdl_stmt ins_patterns)
                          Unification.empty
      | Region.mk _ => do
          IO.println (s!"PDL2U.convert: expected only one BB with no " ++
            "arguments in the pattern region")
          return none
  | Op.mk "pdl.pattern" _ _ _ attrs ty => do
      IO.println (s!"PDL2U.convert: expected operation to have exactly one " ++
        "argument (a region):\n{pattern}")
      return none
  | _ => do
      IO.println s!"PDL2U.convert: not a PDL pattern: {pattern}"
      return none

end PDL2U


/-
### Example PDL program
-/

private def ex_pdl: Op := [mlir_op|
  "pdl.pattern"() ({
    -- %T0 = pdl.type
    %T0 = "pdl.type"() : () -> !"pdl.type"
    -- %T1 = pdl.type: i32
    %T1 = "pdl.type"() {type = i32} : () -> !"pdl.type"
    -- %v2 = pdl.operand
    %v2 = "pdl.operand"() : () -> !"pdl.value"
    -- %O3 = pdl.operation "foo.op1"(%v2) -> %T0
    %O3 = "pdl.operation"(%v2, %T0) {attributeNames = [], name = "foo.op1", operand_segment_sizes = dense<[1, 0, 1]> : vector<x3:i32>} : (!"pdl.value", !"pdl.type") -> !"pdl.operation"
    -- %v4 = pdl.result 0 of %O3
    %v4 = "pdl.result"(%O3) {index = 0 : i32} : (!"pdl.operation") -> !"pdl.value"
    -- %v5 = pdl.operand: %T0
    %v5 = "pdl.operand"(%T0) : (!"pdl.type") -> !"pdl.value"
    -- %O6 = pdl.operation "foo.op2"(%v4, %v5) -> %T1
    %O6 = "pdl.operation"(%v4, %v5, %T1) {attributeNames = [], name = "foo.op2", operand_segment_sizes = dense<[2, 0, 1]> : vector<x3:i32>} : (!"pdl.value", !"pdl.value", !"pdl.type") -> !"pdl.operation"

    -- TODO
    "pdl.rewrite"(%O6) ({
      "pdl.replace"(%O6, %v2) {operand_segment_sizes = dense<[1, 0, 1]> : vector<x3:i32>} : (!"pdl.operation", !"pdl.value") -> ()
    }) {operand_segment_sizes = dense<[1, 0]> : vector<x2:i32>} : (!"pdl.operation") -> ()
  }) {benefit = 1 : i16} : () -> ()
]

private def foo_op1_pattern: UTerm :=
  KnownOp "foo.op1"
    [(Var "x", Var "T")]
    [(Var "res", Var "T")]

#print foo_op1_pattern
private def foo_op2_pattern: UTerm :=
  KnownOp "foo.op2"
    [(Var "x", Var "T"), (Var "y", ConstType (MLIRTy.int 32))]
    [(Var "res", Var "T")]

private def foo_dialect := [foo_op1_pattern, foo_op2_pattern]

#eval show IO Unit from do
  let u? ← PDL2U.convert ex_pdl foo_dialect
  match u? with
  | some u =>
      IO.println $ "\nFinal problem:\n--------------\n" ++ u.str_full
      let u ← u.solve 999
      -- let stmt := u.get!.apply ex_root
      -- IO.println s!"Theorem input:\n{stmt}"
  | none =>
      IO.println "none"
