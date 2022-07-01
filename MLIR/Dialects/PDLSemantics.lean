/-
## Rewriting MLIR fragments with PDL

This file implements some framework support for PDL rewrites, which is used to
formalize and prove rewrite operations. We interpret the matching section of
PDL rewrites with the framework's basic pattern-matching tools, which allows us
to unify (enrich) them with constraints specified by the dialect. This is
important because we later need to reason on the semantics of matched programs,
which are only defined if the constraints are met.

The end goal is to output a template theorem that looks like this:

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

Operations of PDL that we support right now:

* `pdl.operand`:
  Used as unification variable. Supports type requirement.
* `pdl.operation`:
  Used as unification term.
* `pdl.pattern`:
  Input operation for the whole rewriting workflow.
* `pdl.result`:
  Utility to reference results from `pdl.operation` pointers
* `pdl.type`:
  Used as unification variable.

Operations of PDL that we should partially or totally express:

* `pdl.attribute`:
  We don't have attributes yet
* `pdl.erase`:
  We don't compute the target yet
* `pdl.operation`:
  We should support attributes. And we don't handle the output either
* `pdl.replace`:
  We don't compute the target yet
* `pdl.rewrite`:
  One of the main components of the specification. We might not care about the
  root selection since we don't implement the pattern matching/search.

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

import MLIR.Semantics.Types
import MLIR.Semantics.Matching
import MLIR.Semantics.Unification
import MLIR.AST
import MLIR.EDSL
import Lean.Exception
import Lean.Elab.Term
open Lean

open MLIR.AST
open MLIR.EDSL

-- TODO: We updated to new matching but did not add regular type checks

/- Utilities -/

private def MLIR.AST.SSAVal.str: MLIR.AST.SSAVal → String
  | SSAVal.SSAVal s => s

-- Get the n-th category of arguments from the specified variadic list by using
-- the operand segment size attribute
private def operandSegment (args: List SSAVal) (oss: TensorElem) (n: Nat) :=
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


/-
### Monad for the analysis of PDL programs

While PDL programs are mostly similar to match terms, the translation involves
some bookkeeping and fresh name generation. The Translation structure keeps
track of this information, which is carried around in a state monad called
TranslationM.
-/

structure Translation (δ: Dialect α σ ε) where mk ::
  -- Unification problem
  u: Unification δ
  -- All variables defined so far (for fresh name generation)
  allvars: List String
  -- Typing judgements (to be inserted into operation terms)
  judgements: List (String × MTerm δ)
  -- List of operations that we want to collect after unification
  operations: List (MTerm δ)
  -- Names of results for each operation, and their types
  opresults: List (String × List (String × MTerm δ))
  -- Whether translation completed successfully
  success: Bool

def Translation.empty: Translation δ :=
  { u           := Unification.empty,
    allvars     := [],
    judgements  := [],
    operations  := [],
    opresults   := [],
    success     := false }

instance {δ: Dialect α σ ε} : Inhabited (Translation δ) := ⟨Translation.empty⟩

def Translation.str: Translation δ → String := fun tr =>
  "Unification problem:\n" ++
    (toString tr.u) ++
  "\nAll variables:\n " ++
    (String.join $ tr.allvars.map (s!" %{·}")) ++
  "\nTyping judgements:\n" ++
    (String.join $ tr.judgements.map (fun (v,t) => s!"  %{v}: {t}\n")) ++
  "Operation results:\n" ++
    (String.join $ tr.opresults.map (fun (n,l) => s!"  %{n}: {l}\n"))

abbrev TranslationM (δ: Dialect α σ ε) := StateT (Translation δ) IO

def TranslationM.toIO {α} (tr: Translation δ) (x: TranslationM δ α): IO α :=
  Prod.fst <$> StateT.run x tr

def TranslationM.error {α} (s: String): TranslationM δ α :=
  throw <| IO.userError (s ++ " o(x_x)o")

/-
#### Name recording and name generation

The following monad functions are concerned with tracking variables,
guaranteeing uniqueness, and generating fresh names.
-/

-- Record that [name] is now used in the problem, and check uniqueness
def TranslationM.addName (name: String): TranslationM δ Unit := do
  let tr ← get
  if name ∈ tr.allvars then
    error s!"addName: {name} is already used!"
  else
    set { tr with allvars := tr.allvars ++ [name] }

-- Check that [name] is known
def TranslationM.checkNameDefined (name: String): TranslationM δ Unit := do
  let tr ← get
  if ! name ∈ tr.allvars then
    error s!"checkNameDefined: {name} is undefined!"

-- Make up to [n] attempts at finding a fresh name by suffixing [s]
private def TranslationM.freshNameAux (s: String) (n p: Nat):
    TranslationM δ String := do
  let tr ← get
  match n with
  | 0 =>
      return s
  | m+1 =>
      let s' := s!"{s}{p}"
      if tr.allvars.all (· != s') then
        set { tr with allvars := tr.allvars ++ [s'] }
        return s'
      else
        freshNameAux s m (p+1)

-- Generate a new fresh name based on [name]; if not available, resort to
-- adding numbered suffixes
def TranslationM.makeFreshName (name: String): TranslationM δ String := do
  let tr ← get
  if tr.allvars.all (· != name) then
    addName name
    return name
  else
    let f ← freshNameAux name (tr.allvars.length+1) 0
    addName f
    return f

-- Generate [n] fresh names based on [name]
def TranslationM.makeFreshNames (name: String) (n: Nat):
    TranslationM δ (List String) :=
  (List.range n).mapM (fun idx => makeFreshName (name ++ toString idx))

-- Generate a copy of the operation with the specified prefix and fresh names.
-- Returns a pair with the new term and the list of all variables involved.
def TranslationM.makeFreshOp (prefix_: String) (op: MTerm δ) (priority: Nat):
    TranslationM δ (MTerm δ) := do
  let renameVars (done: List String) (vars: List (String × MSort)):
      TranslationM δ (List (String × MTerm δ) × List String) :=
    vars.foldlM
      (fun (repl, done) (var, sort) => do
        if var ∈ done then
          return (repl, done)
        else
          let var' ← makeFreshName (prefix_ ++ var)
          return (repl ++ [(var, MTerm.Var priority var' sort)], done ++ [var]))
      ([], done)

  let (repl, done) ← renameVars [] op.varsWithSorts
  return op.substVars repl


/-
#### Access to translation data

The following utilities query data recorded in the translation state.
-/

def TranslationM.addEquation (equation: UEq δ): TranslationM δ Unit := do
  let tr ← get
  set { tr with u := { equations := tr.u.equations ++ [equation] } }

def TranslationM.addOperation (op: MTerm δ): TranslationM δ Unit := do
  let tr ← get
  set { tr with operations := tr.operations ++ [op] }

def TranslationM.addJudgement (s: String) (type: MTerm δ): TranslationM δ Unit := do
  let tr ← get
  set { tr with judgements := tr.judgements ++ [(s, type)] }

def TranslationM.addOpResults (name: String) (results: List (String × MTerm δ)):
    TranslationM δ Unit := do
  let tr ← get
  set { tr with opresults := tr.opresults ++ [(name, results)] }


def TranslationM.findJudgement? (s: String): TranslationM δ (Option (MTerm δ)) := do
  let cmpName := fun (name, type) => if name = s then some type else none
  return (← get).judgements.findSome? cmpName

def TranslationM.findJudgement (s: String): TranslationM δ (MTerm δ) := do
  match ← findJudgement? s with
  | some type   => return type
  | none        => error s!"findJudgement: no type information for {s}!"

def TranslationM.findJudgements (l: List String):
    TranslationM δ (List (String × MTerm δ)) :=
  l.mapM (fun var => do return (var, ← findJudgement var))

def TranslationM.findOpResult? (op: String) (idx: Nat):
    TranslationM δ (Option (String × MTerm δ)) := do
  let cmpName := fun (name, rets) => if name = op then rets.get? idx else none
  return (← get).opresults.findSome? cmpName

def TranslationM.findOpResult (op: String) (idx: Nat):
    TranslationM δ (String × MTerm δ) := do
  match ← findOpResult? op idx with
  | some info   => return info
  | none        => error s!"findOpResultName: no return #{idx} for {op}!"


/-
### Translation of PDL statements to match problems

Interpretation PDL programs as match problems is fairly straightforward; most
PDL operations simply add new names or equations to the. Most of the work is
spent on parsing the input, adding new names and keeping track of information.

PDL has a lot of restrictions on what you can write; you can't use the same
variable at two different places (implicit unification), you can't have a
declaration operation (pdl.value, pdl.type, etc) without binding it to a
variable, etc. This allows us to make assumptions on the shape of the input.
-/

section
open TranslationM

-- TODO: Provide dialect data properly once implemented
private def PDLToMatch.readStatement (operationMatchTerms: List (MTerm builtin))
    (stmt: BasicBlockStmt builtin): TranslationM builtin Unit := do
  let tr ← get

  match stmt with
  | BasicBlockStmt.StmtAssign (SSAVal.SSAVal name) _ op => match op with

    -- %name = pdl.type
    | Op.mk "pdl.type" [] [] [] (AttrDict.mk [])
          (.fn (.tuple []) (.undefined "pdl.type")) => do
        IO.println s!"Found new type variable: {name}"
        addName name

    -- %name = pdl.type: TYPE
    | Op.mk "pdl.type" [] [] [] (AttrDict.mk [
            AttrEntry.mk "type" (AttrValue.type τ)
          ])
          (.fn (.tuple []) (.undefined "pdl.type")) => do
        IO.println s!"Found new type variable: {name} (= {τ})"
        addName name
        addEquation (.Var 1 name .MMLIRType, .ConstMLIRType τ)

    -- %name = pdl.operand
    | Op.mk "pdl.operand" [] [] [] (AttrDict.mk [])
          (.fn (.tuple []) (.undefined "pdl.value")) => do
        IO.println s!"Found new variable: {name}"
        addName name
        let typeName ← makeFreshName (name ++ "_T")
        IO.println s!"→ Generated type name: {typeName}"
        addJudgement name (.Var 1 typeName .MMLIRType)

    -- %name = pdl.operand: %typeName
    | Op.mk "pdl.operand" [(SSAVal.SSAVal typeName)] [] [] (AttrDict.mk [])
          (.fn (.tuple [.undefined "pdl.type"])
               (.undefined "pdl.value")) => do
        IO.println s!"Found new variable: {name} of type {typeName}"
        addName name
        checkNameDefined typeName
        addJudgement name (.Var 0 typeName .MMLIRType)

    -- %name = pdl.operation "OPNAME"(ARGS) -> TYPE
    | Op.mk "pdl.operation" args [] [] attrs some_type => do
        let (attributeNames, opname, operand_segment_sizes) :=
          (attrs.find "attributeNames",
           attrs.find "name",
           attrs.find "operand_segment_sizes")

        match attributeNames, opname, operand_segment_sizes with
        | some (.list attributeNames),
          some (.str opname),
          some (builtin.dense_vector_attr oss _ _ _) =>
            let values := (operandSegment args oss 0).map (·.str)
            let types  := (operandSegment args oss 2).map (·.str)
            IO.println s!"Found new operation: {name} matching {opname}"
            addName name

            IO.println s!"→ Arguments: {values}, return types: {types}"
            values.forM checkNameDefined
            types.forM checkNameDefined

            let valuesTypes ← findJudgements values
            let retNames ← makeFreshNames (name ++ "_res") types.length
            IO.println s!"→ Arg types: {valuesTypes}, return names: {retNames}"

            let insPattern := operationMatchTerms.find? fun
              | .App .OP [.ConstString n, _, _] => n = opname
              | _ => false
            if insPattern.isNone then
              error s!"pdl.operation: no pattern known for {opname}!"
            let insPattern := insPattern.get!

            IO.println s!"→ Using match term: {insPattern}"
            let ins ← makeFreshOp (name ++ "_") insPattern 2
            addOperation ins
            IO.println s!"→ Instantiated match term: {ins}"

            let operands_args: List (MTerm _) := valuesTypes.map (fun (v,t) =>
              .App .OPERAND [.Var 0 v .MSSAVal, t])
            let operands_rets: List (MTerm _) :=
              List.zip retNames types |>.map (fun (v, t) =>
                .App .OPERAND [.Var 1 v .MSSAVal, .Var 0 t .MMLIRType])
            let op_mterm :=
              .App .OP [
                .ConstString opname,
                .App (.LIST .MOperand) operands_args,
                .App (.LIST .MOperand) operands_rets
              ]
            addEquation (op_mterm, ins)

            let opResults := List.zip retNames (types.map (.Var 0 · .MMLIRType))
            addOpResults name opResults

        | _, _, _ =>
            error s!"pdl.operation: unexpected attributes: {attrs}"

    -- %name = pdl.result INDEX of %op
    | Op.mk "pdl.result" [SSAVal.SSAVal opname] [] [] attrs
          (.fn (.tuple [.undefined "pdl.operation"])
               (.undefined "pdl.value")) => do

        match attrs.find "index" with
        | some (AttrValue.int index (MLIRType.int _ _)) =>
            IO.println
              s!"Found new variable: {name} aliasing result {index} of {opname}"
            checkNameDefined opname
            addName name

            let (resName, resType) ← findOpResult opname index.toNat
            addJudgement name resType
            addEquation (.Var 0 name .MSSAVal, .Var 1 resName .MSSAVal)

        | _ =>
            error s!"pdl.result: unexpected attributes on {opname}: {attrs}"

    | _ => do
        error s!"{op.name}: unrecognized PDL operation"

  | BasicBlockStmt.StmtOp op => match op with
    -- TODO: pdl.rewrite
    | Op.mk "pdl.rewrite" args bbs regions attrs ty =>
        return ()
    | _ => do
        error s!"{op.name}: unrecognized PDL operation"

def PDLToMatch.convert (PDLProgram: Op builtin)
    (operationMatchTerms: List (MTerm builtin)):
    TranslationM builtin Unit :=
  match PDLProgram with
  | Op.mk "pdl.pattern" [] [] [region] attrs ty =>
      match region with
      | Region.mk [BasicBlock.mk name [] stmts] => do
          stmts.forM (readStatement operationMatchTerms)
          set { ← get with success := true }
      | Region.mk _ => do
          error (s!"PDLToMatch.convert: expected only one BB with no " ++
            "arguments in the pattern region")
  | Op.mk "pdl.pattern" _ _ _ attrs ty => do
      error (s!"PDLToMatch.convert: expected operation to have exactly one " ++
        "argument (a region):\n{pattern}")
  | _ => do
      error s!"PDLToMatch.convert: not a PDL program: {PDLProgram}"

def PDLToMatch.unify: TranslationM δ Unit := do
  let tr ← get
  if ! tr.success then
    error "unify: translation did not complete successfully"
  if let some u_unified ← tr.u.solve then
    set { tr with u := u_unified }
  else
    error "unify: unification failed"

def PDLToMatch.getOperationMatchTerms: TranslationM δ (List (MTerm δ)) := do
  let tr ← get
  return tr.operations.map tr.u.applyOnTerm

end

/-
### Example PDL program
-/

private def ex_pdl: Op builtin := [mlir_op|
  "pdl.pattern"() ({
    -- %T0 = pdl.type
    %T0 = "pdl.type"() : () -> !"pdl.type"
    -- %T1 = pdl.type: i32
    %T1 = "pdl.type"() {type = i32} : () -> !"pdl.type"
    -- %v2 = pdl.operand
    %v2 = "pdl.operand"() : () -> !"pdl.value"
    -- %O3 = pdl.operation "foo.op1"(%v2) -> %T0
    %O3 = "pdl.operation"(%v2, %T0) {attributeNames = [], name = "foo.op1", operand_segment_sizes = dense<[1, 0, 1]> : vector<3×i32>} : (!"pdl.value", !"pdl.type") -> !"pdl.operation"
    -- %v4 = pdl.result 0 of %O3
    %v4 = "pdl.result"(%O3) {index = 0 : i32} : (!"pdl.operation") -> !"pdl.value"
    -- %v5 = pdl.operand: %T0
    %v5 = "pdl.operand"(%T0) : (!"pdl.type") -> !"pdl.value"
    -- %O6 = pdl.operation "foo.op2"(%v4, %v5) -> %T1
    %O6 = "pdl.operation"(%v4, %v5, %T1) {attributeNames = [], name = "foo.op2", operand_segment_sizes = dense<[2, 0, 1]> : vector<3×i32>} : (!"pdl.value", !"pdl.value", !"pdl.type") -> !"pdl.operation"

    -- TODO
    "pdl.rewrite"(%O6) ({
      "pdl.replace"(%O6, %v2) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3×i32>} : (!"pdl.operation", !"pdl.value") -> ()
    }) {operand_segment_sizes = dense<[1, 0]> : vector<2×i32>} : (!"pdl.operation") -> ()
  }) {benefit = 1 : i16} : () -> ()
]

-- %res:!T = "foo.op1"(%x:!T)
private def foo_op1_pattern: MTerm builtin :=
  .App .OP [
    .ConstString "foo.op1",
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 1 "x" .MSSAVal, .Var 1 "T" .MMLIRType]],
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 1 "ret" .MSSAVal, .Var 1 "T" .MMLIRType]]
  ]
#eval foo_op1_pattern

-- %res:!T = "foo.op2"(%x:!T, %y:i32)
private def foo_op2_pattern: MTerm builtin :=
  .App .OP [
    .ConstString "foo.op2",
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 1 "x" .MSSAVal, .Var 1 "T" .MMLIRType],
      .App .OPERAND [.Var 1 "y" .MSSAVal, .ConstMLIRType .i32]],
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 1 "ret" .MSSAVal, .Var 1 "T" .MMLIRType]]
  ]
#eval foo_op2_pattern

private def foo_dialect := [foo_op1_pattern, foo_op2_pattern]

#eval show IO Unit from do
  TranslationM.toIO Translation.empty do
    PDLToMatch.convert ex_pdl foo_dialect
    IO.println $ "\n## Translation result ##\n\n" ++ (← get).str
    PDLToMatch.unify
    let matchTerms ← PDLToMatch.getOperationMatchTerms
    IO.println "\n## Final operation match terms ##\n"
    IO.println $ "\n".intercalate (matchTerms.map toString)
