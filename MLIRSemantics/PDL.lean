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

import MLIRSemantics.Types
import MLIRSemantics.Matching
import MLIRSemantics.Unification
import MLIR.AST
import MLIR.EDSL
import Lean.Exception
import Lean.Elab.Term
open Lean

open MLIR.AST
open MLIR.EDSL
open MValue MType MOp

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

structure Translation where mk ::
  -- Unification problem
  u: Unification
  -- All variables defined so far (for fresh name generation)
  allvars: List String
  -- Typing judgements (to be inserted into operation terms)
  judgements: List (String × MType)
  -- List of operations that we want to collect after unification
  operations: List MOp
  -- Names of results for each operation, and their types
  opresults: List (String × List (String × MType))
  -- Whether translation completed successfully
  success: Bool

def Translation.empty: Translation :=
  { u           := Unification.empty,
    allvars     := [],
    judgements  := [],
    operations  := [],
    opresults   := [],
    success     := false }

instance: Inhabited Translation := ⟨Translation.empty⟩

def Translation.str: Translation → String := fun tr =>
  "Unification problem:\n" ++
    (toString tr.u) ++
  "All variables:\n " ++
    (String.join $ tr.allvars.map (s!" %{·}")) ++
  "\nTyping judgements:\n" ++
    (String.join $ tr.judgements.map (fun (v,t) => s!"  %{v}: {t}\n")) ++
  "Operation results:\n" ++
    (String.join $ tr.opresults.map (fun (n,l) => s!"  %{n}: {l}\n"))

abbrev TranslationM := StateT Translation IO

def TranslationM.toIO {α} (tr: Translation) (x: TranslationM α): IO α :=
  Prod.fst <$> StateT.run x tr

def TranslationM.error {α} (s: String): TranslationM α :=
  throw <| IO.userError (s ++ " o(x_x)o")

/-
#### Name recording and name generation

The following monad functions are concerned with tracking variables,
guaranteeing uniqueness, and generating fresh names.
-/

-- Record that [name] is now used in the problem, and check uniqueness
def TranslationM.addName (name: String): TranslationM Unit := do
  let tr ← get
  if name ∈ tr.allvars then
    error s!"addName: {name} is already used!"
  else
    set { tr with allvars := tr.allvars ++ [name] }

-- Check that [name] is known
def TranslationM.checkNameDefined (name: String): TranslationM Unit := do
  let tr ← get
  if ! name ∈ tr.allvars then
    error s!"checkNameDefined: {name} is undefined!"

-- Make up to [n] attempts at finding a fresh name by suffixing [s]
private def TranslationM.freshNameAux (s: String) (n p: Nat):
    TranslationM String := do
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
def TranslationM.makeFreshName (name: String): TranslationM String := do
  let tr ← get
  if tr.allvars.all (· != name) then
    return name
  else
    freshNameAux name (tr.allvars.length+1) 0

-- Generate [n] fresh names based on [name]
def TranslationM.makeFreshNames (name: String) (n: Nat):
    TranslationM (List String) :=
  (List.range n).mapM (fun idx => makeFreshName (name ++ toString idx))

-- Generate a copy of the operation with the specified prefix and fresh names.
-- Returns a pair with the new term and the list of all variables involved.
def TranslationM.makeFreshOp (prefix_: String) (op: MOp) (priority: Nat):
    TranslationM MOp := do
  let renameVars {α} (done vars: List String) (ctor: String → α):
      TranslationM (List (String × α) × List String) :=
    vars.foldlM
      (fun (repl, done) var => do
        if var ∈ done then
          return (repl, done)
        else
          let var' ← makeFreshName (prefix_ ++ var)
          return (repl ++ [(var, ctor var')], done ++ [var]))
      ([], done)

  let (replValues, done₁) ← renameVars [] op.valueVars (ValueVar priority)
  let (replTypes,  done₂) ← renameVars [] op.typeVars (TypeVar priority)
  return (op.substValues replValues).substTypes replTypes


/-
#### Access to translation data

The following utilities query data recorded in the translation state.
-/

def TranslationM.addEquation (equation: UEq): TranslationM Unit := do
  let tr ← get
  set { tr with u := { equations := tr.u.equations ++ [equation] } }

def TranslationM.addOperation (op: MOp): TranslationM Unit := do
  let tr ← get
  set { tr with operations := tr.operations ++ [op] }

def TranslationM.addJudgement (s: String) (type: MType): TranslationM Unit := do
  let tr ← get
  set { tr with judgements := tr.judgements ++ [(s, type)] }

def TranslationM.addOpResults (name: String) (results: List (String × MType)):
    TranslationM Unit := do
  let tr ← get
  set { tr with opresults := tr.opresults ++ [(name, results)] }


def TranslationM.findJudgement? (s: String): TranslationM (Option MType) := do
  let cmpName := fun (name, type) => if name = s then some type else none
  return (← get).judgements.findSome? cmpName

def TranslationM.findJudgement (s: String): TranslationM MType := do
  match ← findJudgement? s with
  | some type   => return type
  | none        => error s!"findJudgement: no type information for {s}!"

def TranslationM.findJudgements (l: List String):
    TranslationM (List (String × MType)) :=
  l.mapM (fun var => do return (var, ← findJudgement var))

def TranslationM.findOpResult? (op: String) (idx: Nat):
    TranslationM (Option (String × MType)) := do
  let cmpName := fun (name, rets) => if name = op then rets.get? idx else none
  return (← get).opresults.findSome? cmpName

def TranslationM.findOpResult (op: String) (idx: Nat):
    TranslationM (String × MType) := do
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
-- TODO: Set name substitution priorities instead of using the default
private def PDLToMatch.readStatement (operationMatchTerms: List MOp)
    (stmt: BasicBlockStmt): TranslationM Unit := do
  let tr ← get

  match stmt with
  | BasicBlockStmt.StmtAssign (SSAVal.SSAVal name) _ op => match op with

    -- %name = pdl.type
    | Op.mk "pdl.type" [] [] [] (AttrDict.mk [])
          (MLIRTy.fn (MLIRTy.tuple [])
                     (MLIRTy.user "pdl.type")) => do
        IO.println s!"Found new type variable: {name}"
        addName name

    -- %name = pdl.type: TYPE
    | Op.mk "pdl.type" [] [] [] (AttrDict.mk [
            AttrEntry.mk "type" (AttrVal.type τ)
          ])
          (MLIRTy.fn (MLIRTy.tuple [])
                     (MLIRTy.user "pdl.type")) => do
        IO.println s!"Found new type variable: {name} (= {τ})"
        addName name
        addEquation <| .EqType (TypeVar 1 name) (TypeConst τ)

    -- %name = pdl.operand
    | Op.mk "pdl.operand" [] [] [] (AttrDict.mk [])
          (MLIRTy.fn (MLIRTy.tuple [])
                     (MLIRTy.user "pdl.value")) => do
        IO.println s!"Found new variable: {name}"
        addName name
        let typeName ← makeFreshName (name ++ "_T")
        IO.println s!"→ Generated type name: {typeName}"
        addJudgement name (TypeVar 1 typeName)

    -- %name = pdl.operand: %typeName
    | Op.mk "pdl.operand" [(SSAVal.SSAVal typeName)] [] [] (AttrDict.mk [])
          (MLIRTy.fn (MLIRTy.tuple [MLIRTy.user "pdl.type"])
                     (MLIRTy.user "pdl.value")) => do
        IO.println s!"Found new variable: {name} of type {typeName}"
        addName name
        checkNameDefined typeName
        addJudgement name (TypeVar 0 typeName)

    -- %name = pdl.operation "OPNAME"(ARGS) -> TYPE
    | Op.mk "pdl.operation" args [] [] attrs some_type => do
        let (attributeNames, opname, operand_segment_sizes) :=
          (attrs.find "attributeNames",
           attrs.find "name",
           attrs.find "operand_segment_sizes")

        match attributeNames, opname, operand_segment_sizes with
        | some (AttrVal.list attributeNames),
          some (AttrVal.str opname),
          some (AttrVal.dense oss (MLIRTy.vector _ _ _)) =>
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

            let insPattern := operationMatchTerms.find? fun t => match t with
              | OpKnown n _ _ => n = opname
            if insPattern.isNone then
              error s!"pdl.operation: no pattern known for {opname}!"
            let insPattern := insPattern.get!

            IO.println s!"→ Using match term: {insPattern}"
            let ins ← makeFreshOp (name ++ "_") insPattern 2
            addOperation ins
            IO.println s!"→ Instantiated match term: {ins}"

            let args := valuesTypes.map (fun (v,t) => (ValueVar 0 v, t))
            let rets := List.zip (retNames.map (ValueVar 1)) (types.map TypeVar)
            addEquation <| .EqOp (OpKnown opname args rets) ins

            let opResults := List.zip retNames (types.map TypeVar)
            addOpResults name opResults

        | _, _, _ =>
            error s!"pdl.operation: unexpected attributes: {attrs}"

    -- %name = pdl.result INDEX of %op
    | Op.mk "pdl.result" [SSAVal.SSAVal opname] [] [] attrs
          (MLIRTy.fn (MLIRTy.tuple [MLIRTy.user "pdl.operation"])
                     (MLIRTy.user "pdl.value")) => do

        match attrs.find "index" with
        | some (AttrVal.int index (MLIRTy.int _)) =>
            IO.println
              s!"Found new variable: {name} aliasing result {index} of {opname}"
            checkNameDefined opname
            addName name

            let (resName, resType) ← findOpResult opname index.toNat
            addJudgement name resType
            addEquation <| .EqValue (ValueVar 0 name) (ValueVar 1 resName)

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

def PDLToMatch.convert (PDLProgram: Op) (operationMatchTerms: List MOp):
    TranslationM Unit :=
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

def PDLToMatch.unify: TranslationM Unit := do
  let tr ← get
  if ! tr.success then
    error "unify: translation did not complete successfully"
  if let some u_unified ← tr.u.solve then
    set { tr with u := u_unified }
  else
    error "unify: unification failed"

def PDLToMatch.getOperationMatchTerms: TranslationM (List MOp) := do
  let tr ← get
  return tr.operations.map tr.u.applyOnOp

end

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

private def foo_op1_pattern: MOp :=
  OpKnown "foo.op1"
    [(ValueVar 1 "x", TypeVar 1 "T")]
    [(ValueVar 1 "res", TypeVar 1 "T")]
#eval foo_op1_pattern

private def foo_op2_pattern: MOp :=
  OpKnown "foo.op2"
    [(ValueVar 1 "x", TypeVar 1 "T"),
     (ValueVar 1 "y", TypeConst (MLIRTy.int 32))]
    [(ValueVar 1 "res", TypeVar 1 "T")]
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
