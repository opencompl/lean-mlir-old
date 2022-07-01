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
import MLIR.EDSL
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

@[reducible]
def MSort.toType (δ: Dialect α σ ε): MSort -> Type
| .MOp => Bool -- TODO MLIR.AST.BasicBlockStmt δ
| .MOperand => MLIR.AST.SSAVal × MLIR.AST.MLIRType δ
| .MMLIRType => MLIR.AST.MLIRType δ
| .MSSAVal => MLIR.AST.SSAVal
| .MAttrValue => Bool -- TODO MLIR.AST.AttrValue δ
| .MNat => Nat
| .MString => String
| .MDimension => Dimension
| .MSignedness => MLIR.AST.Signedness
| .MList mTerm => List (mTerm.toType δ)

def MSort_toType_decEq {δ: Dialect α σ ε} (s: MSort): DecidableEq (s.toType δ) :=
  match s with
  | .MOp => decEq
  | .MOperand => decEq 
  | .MMLIRType => decEq
  | .MSSAVal => decEq
  | .MAttrValue => decEq
  | .MNat => decEq
  | .MString => decEq
  | .MDimension => decEq
  | .MSignedness => decEq
  | .MList term => @List.hasDecEq _ (MSort_toType_decEq term)

instance {δ: Dialect α σ ε} (s: MSort): DecidableEq (s.toType δ) := MSort_toType_decEq s

/-
### Variable context for MTerms

This structure contains an assignment from MTerm variables to concrete structures.
It is used both for matching, and for concretizing MTerms into concrete strucctures.
-/

-- Matching context. Contains the assignment of matching variables.
abbrev VarCtx (δ: Dialect α σ ε) := List ((s: MSort) × List (String × (s.toType δ)))

-- Get the assignment of a variable.
def VarCtx.get (ctx: VarCtx δ) (s: MSort) (name: String) : Option (s.toType δ) :=
  match ctx with
  | {fst := so, snd := sortCtx}::ctx' => 
    match H: so == s with
    | false => get ctx' s name 
    | true => (List.find? (·.fst == name) ((of_decide_eq_true H) ▸ sortCtx)).map (·.snd)
  | [] => none

-- Assign a variable.
def VarCtx.set (ctx: VarCtx δ) (s: MSort) (name: String) (value: s.toType δ) : VarCtx δ :=
  match ctx with
  | {fst := so, snd := sortCtx}::ctx' => 
    match H: so == s with
    | false => {fst := so, snd := sortCtx}::(set ctx' s name value) 
    | true => {fst := so, snd := (name, (of_decide_eq_true H) ▸ value)::sortCtx}::ctx' 
  | [] => [{fst := s, snd := [(name, value)]}]

/-
### Concretization of MTerm

This section defines some functions to transform a MTerm into some
concrete structure, given a variable context.
-/

-- We provide an expected sort, since we do not want to carry the
-- proof that terms are well typed.
def MTerm.concretizeVariable (m: MTerm) (s: MSort) (ctx: VarCtx δ) : Option (s.toType δ) :=
  match m with
  | Var _ name sort =>
    if s == sort then ctx.get s name else none
  | _ => none

def MTerm.concretizeSign (m: MTerm) (ctx: VarCtx δ) : Option Signedness := 
  match m with
  | Var _ _ _ => m.concretizeVariable .MSignedness ctx
  | ConstSignedness s => some s
  | _ => none

def MTerm.concretizeNat (m: MTerm) (ctx: VarCtx δ) : Option Nat := 
  match m with
  | Var _ _ _ => m.concretizeVariable .MNat ctx
  | ConstNat n => some n
  | _ => none

def MTerm.concretizeDim (m: MTerm) (ctx: VarCtx δ) : Option Dimension := 
  match m with
  | Var _ _ _ => m.concretizeVariable .MDimension ctx
  | ConstDimension d => some d
  | _ => none

def MTerm.concretizeType (m: MTerm) (ctx: VarCtx δ) : Option (MLIR.AST.MLIRType δ) :=
  match m with
  | Var _ _ _ => m.concretizeVariable .MMLIRType ctx
  | .App .INT [mSign, mNat] => do
    let sign ← mSign.concretizeSign ctx
    let nat ← mNat.concretizeNat ctx
    return MLIRType.int sign nat
  | _ => none

def MTerm.concretizeOperand (m: MTerm) (ctx: VarCtx δ) : Option (MLIR.AST.SSAVal × MLIR.AST.MLIRType δ) :=
  match m with
  | Var _ _ _ => m.concretizeVariable .MOperand ctx 
  | .App .OPERAND [mVal, mType] => do
    let val ← mVal.concretizeVariable .MSSAVal ctx
    let type ← mType.concretizeType ctx
    return (val, type)
  | _ => none

def MTerm.concretizeOperands (m: MTerm) (ctx: VarCtx δ) : Option (List (MLIR.AST.SSAVal × MLIR.AST.MLIRType δ)) :=
  match m with
  | .App (.LIST .MOperand) l => l.mapM (fun m' => m'.concretizeOperand ctx)
  | _ => none

def MTerm.concretizeOp (m: MTerm) (ctx: VarCtx δ) : Option (BasicBlockStmt δ) :=
  match m with
  | .App .OP [ .ConstString mName, mOperands, mRes ] => do
    let operands ← MTerm.concretizeOperands mOperands ctx
    let res ← MTerm.concretizeOperands mRes ctx
    let operandsVal := operands.map (fun p => p.fst)
    let operandsTy := operands.map (fun p => p.snd)
    match res with
    | [(resVal, resTy)] => return .StmtAssign resVal none (.mk mName operandsVal [] [] (AttrDict.mk []) (MLIRType.fn (MLIRType.tuple operandsTy) (MLIRType.tuple [resTy])))
    | _ => none
  | _ => none

def MTerm.concretizeProg (m: List MTerm) (ctx: VarCtx δ) : Option (List (BasicBlockStmt δ)) :=
  m.mapM (fun m' => m'.concretizeOp ctx)

/-
### Simple MTerm matching

This section defines functions to match an MTerm with a concrete structure.
Note that here, the matching does not match recursively inside the concrete structure.
-/

-- Match a MTerm variable.
def matchVariable {δ: Dialect α σ ε} (s: MSort) (name: String) (val: s.toType δ) (ctx: VarCtx δ) : Option (VarCtx δ) := 
  match ctx.get s name with 
  | some matchedVal => if val == matchedVal then some ctx else none
  | none => some (ctx.set s name val)

-- Match a signedness with a MTerm.
def matchMSignedness {δ: Dialect α σ ε} (mSgn: MTerm) (sgn: Signedness)
                     (ctx: VarCtx δ): Option (VarCtx δ) :=
  match mSgn with
  | .Var _ name .MSignedness => matchVariable .MSignedness name sgn ctx
  | .ConstSignedness mSgn => if sgn == mSgn then some ctx else none
  | _ => none

-- Match a dimension with a MTerm.
def matchMDimension {δ: Dialect α σ ε} (mDim: MTerm) (dim: Dimension)
                    (ctx: VarCtx δ): Option (VarCtx δ) :=
  match mDim with
  | .Var _ name .MDimension => matchVariable .MDimension name dim ctx
  | .ConstDimension mDim => if dim == mDim then some ctx else none
  | _ => none

-- Match a string with a MTerm.
def matchMString {δ: Dialect α σ ε} (mStr: MTerm) (str: String)
                 (ctx: VarCtx δ): Option (VarCtx δ) :=
  match mStr with
  | .Var _ name .MString => matchVariable .MString name str ctx
  | .ConstString mStr => if str == mStr then some ctx else none
  | _ => none

-- Match a nat with a MTerm.
def matchMNat {δ: Dialect α σ ε} (mNat: MTerm) (nat: Nat)
              (ctx: VarCtx δ): Option (VarCtx δ) :=
  match mNat with
  | .Var _ name .MNat => matchVariable .MNat name nat ctx
  | .ConstNat mNat => if nat == mNat then some ctx else none
  | _ => none

-- Match a nat with a MTerm.
def matchMType (mType: MTerm) (type: MLIRType builtin)
               (ctx: VarCtx builtin): Option (VarCtx builtin) :=
  match mType, type with
  | .Var _ name .MMLIRType, _ => matchVariable .MMLIRType name type ctx
  | .ConstMLIRType mType, _ => if type == mType then some ctx else none
  | .App .INT [mSgn, mNat], MLIRType.int sgn nat =>
    (matchMSignedness mSgn sgn ctx).bind (matchMNat mNat nat ·)
  | _, _ => none

-- Match a type SSA value with a MTerm.
def matchMSSAVal (mOperand: MTerm) (operand: SSAVal) (operandTy: MLIRType builtin)
                  (ctx: VarCtx builtin) : Option (VarCtx builtin) :=
  match mOperand with
  | .App .OPERAND [.Var _ ssaName .MSSAVal, mType] => 
    (matchMType mType operandTy ctx).bind (matchVariable MSort.MSSAVal ssaName operand ·)
  | _ => none

-- Match a list of typed SSA values with a list of MTerm.
def matchMSSAVals (operands: List SSAVal) (operandsTy: List (MLIRType builtin))
                   (mOperands: List MTerm) (ctx: VarCtx builtin) : Option (VarCtx builtin) :=
  match operands, operandsTy, mOperands with
  | [], [], [] => some ctx
  | operand::operands, operandTy::operandsTy, mOperand::mOperands =>
    (matchMSSAVal mOperand operand operandTy ctx).bind (matchMSSAVals operands operandsTy mOperands ·)
  | _, _, _ => none

-- Match a basic block statement with an MTerm.
def matchMBasicBlockStmt (op: BasicBlockStmt builtin) (mterm: MTerm) (ctx: VarCtx builtin) : Option (VarCtx builtin) :=
  match op, mterm with
  | .StmtAssign res ix (Op.mk name operands [] [] (AttrDict.mk []) (MLIRType.fn (MLIRType.tuple operandsTy) (MLIRType.tuple resultsTy))), 
    .App .OP [ .ConstString mName, .App (.LIST .MOperand) mOperands, .App (.LIST .MOperand) mRes ] =>
    if name != mName then
      none
    else
      (matchMSSAVals operands operandsTy mOperands ctx).bind
      (matchMSSAVals [res] resultsTy mRes ·)
  | _, _ => none

/-
### Recursive MTerm op matching

This section defines functions to match an op MTerm inside a concrete structure.
Here, the matching is done recursively inside the regions/BBs/Ops.

We first define functions that match all possible ops in the IR. Then, we use
this to match a program in an IR.
-/

mutual 
-- Get all possible operations matching an MTerm in a basic block statement.
def matchAllMOpInBBStmt (stmt: BasicBlockStmt builtin) (mOp: MTerm)
                        (ctx: VarCtx builtin) : List (BasicBlockStmt builtin × VarCtx builtin) :=
  let nextMatches :=
    match stmt with
    | .StmtAssign _ _ op => matchAllMOpInOp op mOp ctx
    | .StmtOp op => matchAllMOpInOp op mOp ctx;
  match matchMBasicBlockStmt stmt mOp ctx with
  | some ctx' => (stmt, ctx')::nextMatches
  | none => nextMatches

-- Get all possible operations matching an MTerm in a list of basic block statements.
def matchAllMOpInBBStmts (stmts: List (BasicBlockStmt builtin)) (mOp: MTerm)
                        (ctx: VarCtx builtin) : List (BasicBlockStmt builtin × VarCtx builtin) :=
  match stmts with
  | stmt::stmts' => (matchAllMOpInBBStmt stmt mOp ctx).append (matchAllMOpInBBStmts stmts' mOp ctx)
  | [] => []

-- Get all possible operations matching an MTerm in a basic block.
def matchAllMOpInBB (bb: BasicBlock builtin) (mOp: MTerm)
                    (ctx: VarCtx builtin) : List (BasicBlockStmt builtin × VarCtx builtin) :=
  match bb with
  | .mk _ _ stmts => matchAllMOpInBBStmts stmts mOp ctx
    -- We can't use the following, because we need Lean to understand the mutual definition terminates
    -- List.foldl List.append [] <| List.map (matchAllMOpInBBStmt · mOp ctx) ops

-- Get all possible operations matching an MTerm in multiple basic blocks.
def matchAllMOpInBBs (bbs: List (BasicBlock builtin)) (mOp: MTerm)
                    (ctx: VarCtx builtin) : List (BasicBlockStmt builtin × VarCtx builtin) :=
  match bbs with
  | bb::bbs' => (matchAllMOpInBB bb mOp ctx).append (matchAllMOpInBBs bbs' mOp ctx)
  | [] => []

-- Get all possible operations matching an MTerm in a region.
def matchAllMOpInRegion (region: Region builtin) (mOp: MTerm)
                        (ctx: VarCtx builtin) : List (BasicBlockStmt builtin × VarCtx builtin) :=
  match region with
  | .mk bbs => matchAllMOpInBBs bbs mOp ctx
    -- List.foldl List.append [] <| List.map (matchAllMOpInBB · mOp ctx) bbs

-- Get all possible operations matching an MTerm in a list of regions.
def matchAllMOpInRegions (regions: List (Region builtin)) (mOp: MTerm)
                         (ctx: VarCtx builtin) : List (BasicBlockStmt builtin × VarCtx builtin) :=
  match regions with
  | region::regions' => (matchAllMOpInRegion region mOp ctx).append (matchAllMOpInRegions regions' mOp ctx)
  | [] => []

-- Get all possible operations matching an MTerm in an operation.
def matchAllMOpInOp (op: Op builtin) (mOp: MTerm)
                    (ctx: VarCtx builtin) : List (BasicBlockStmt builtin × VarCtx builtin) :=
  match op with
  | .mk _ _ _ regions _ _ => matchAllMOpInRegions regions mOp ctx
    -- List.foldl List.append [] <| List.map (matchAllMOpInRegion · mOp ctx) regions
end
termination_by
  matchAllMOpInBBStmt stmt _ _ => sizeOf stmt
  matchAllMOpInBBStmts stmts _ _ => sizeOf stmts
  matchAllMOpInBB bb _ _ => sizeOf bb
  matchAllMOpInBBs bbs _ _ => sizeOf bbs
  matchAllMOpInRegion region _ _ => sizeOf region
  matchAllMOpInRegions regions _ _ => sizeOf regions
  matchAllMOpInOp op _ _ => sizeOf op

mutual
-- Match a program defined by a list of MTerm (one Operation per MTerm) in an operation.
def matchMProgInOp (op: Op builtin) (mOps: List MTerm)
                   (ctx: VarCtx builtin) : Option ((List (BasicBlockStmt builtin)) × VarCtx builtin) :=
  match mOps with
  -- Try all matches of the first operation.
  | mOp::mOps' => 
    matchMProgInOpAux op mOps' (matchAllMOpInOp op mOp ctx)
  | [] => some ([], ctx)

-- Match a program defined by a list of MTerm (one Operation per MTerm) in an operation.
-- `matchOps` correspond to the possible matches of the current MTerm being matched.
def matchMProgInOpAux (op: Op builtin) (mOps: List MTerm)
                      (matchOps: List (BasicBlockStmt builtin × VarCtx builtin))
                      : Option (List (BasicBlockStmt builtin) × VarCtx builtin) :=
  -- For all possible match, we check if we can match the remaining of the program
  -- with the match assignment
  match matchOps with
  | (matchOp, ctx')::matchOps' =>
    match matchMProgInOp op mOps ctx' with
    -- If we do match the remaining of the program, we are finished.
    | some (matchedOps, ctx'') => some (matchOp::matchedOps, ctx'')
    -- Otherwise, we check the next match for the current operation.
    | none => matchMProgInOpAux op mOps matchOps'
  | [] => none
end
termination_by
  matchMProgInOpAux _ mOps matchOps => (mOps, matchOps)
  matchMProgInOp op mOps ctx => (mOps, [])

private def test_addi_multiple_pattern: List MTerm :=
  [.App .OP [
    .ConstString "std.addi",
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "op_x" .MSSAVal, .Var 2 "T" .MMLIRType],
      .App .OPERAND [.Var 2 "op_y" .MSSAVal, .Var 2 "T" .MMLIRType]],
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "op_res" .MSSAVal, .Var 2 "T" .MMLIRType]]
  ],
  .App .OP [
    .ConstString "std.addi",
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "op_res" .MSSAVal, .Var 2 "T" .MMLIRType],
      .App .OPERAND [.Var 2 "op_res" .MSSAVal, .Var 2 "T" .MMLIRType]],
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "op_res2" .MSSAVal, .Var 2 "T" .MMLIRType]]
  ]]

private def multiple_example: Op builtin := [mlir_op|
  "builtin.module"() ({
    ^entry:
    %r2 = "std.addi"(%t2, %t3): (i32, i32) -> (i32)
    %r = "std.addi"(%t0, %t1): (i32, i32) -> (i32)
    %r3 = "std.addi"(%r, %r): (i32, i32) -> (i32)
  }) : ()
]

-- Match an MTerm program in some IR, then concretize
-- the MTerm using the resulting matching context.
def multiple_example_result : Option (List (BasicBlockStmt builtin)) := do
  let (val, ctx) ← matchMProgInOp multiple_example test_addi_multiple_pattern []
  let res ← MTerm.concretizeProg test_addi_multiple_pattern ctx
  val

#eval multiple_example_result
