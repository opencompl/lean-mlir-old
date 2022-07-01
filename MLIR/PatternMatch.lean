import MLIR.AST
import Lean.PrettyPrinter
import Std.Data.AssocList
import Lean.Meta
import MLIR.EDSL
import MLIR.Doc
import MLIR.Dialects.PDL
import MLIR.P
import MLIR.MLIRParser
import MLIR.Dialects.BuiltinModel

open MLIR.Doc
open Lean.PrettyPrinter
open Lean.Meta
open Std
open Std.AssocList
open MLIR.AST
open MLIR.EDSL
open MLIR.P
open MLIR.MLIRParser

-- PDL: Pattern Description Language
-- |
-- +-> parse it into MLIR data structure --(lean code that spits out a matcher)
-- |
-- +-> matcher
-- | TODO: for proof automation, it's important that
-- built is the first constructor. Write custom tactics to avoid this.
inductive matcher where
| built: matcher
| kind?: (kind: String) -> matcher -> matcher
| arg?: (ix: Int) -> (name: String) -> matcher -> matcher
| focus!: (name: String) -> matcher -> matcher
| root: matcher -> matcher


-- ^ replace operand of focused op with new op.

-- abbrev Set (α : Type) (compare: α -> α -> Ordering) := RBMap α Unit compare
--
-- instance {α: Type} {β: Type} {compare: α -> α -> Ordering} :
--   Inhabited (RBMap α β compare) where
--   default := RBMap.empty
--
-- def set_union {α: Type} {compare:α -> α -> Ordering} (xs: Set α compare) (ys: Set α compare): Set α compare :=
--   RBMap.fromList (xs.toList ++ ys.toList) compare
--
-- def set_insert{α: Type} {compare:α -> α -> Ordering}
--   (xs: Set α compare) (a: α): Set α compare :=
--   RBMap.insert xs a ()
--
--
-- def RBMap.set_bind {α: Type} {compare: α -> α -> Ordering} (xs: Set α compare) (f: α -> Set α compare): Set α compare :=
--   RBMap.fold (fun new a () => set_union new (f a)) RBMap.empty xs
--
-- def set_map {α: Type} {compare: α -> α -> Ordering} (xs: Set α compare) (f: α -> β) (compare': β -> β -> Ordering): Set β compare' :=
--     RBMap.fold (fun new a () => RBMap.insert new (f a) ()) RBMap.empty xs
--
-- def set_subtract {α: Type} {compare: α -> α -> Ordering} (all: Set α compare) (to_remove: Set α compare): Set α compare :=
--   RBMap.fold (fun all' a () => RBMap.erase all' a) to_remove all
--
-- def set_singleton {α: Type} (a: α) (compare: α -> α -> Ordering): Set α compare :=
--     RBMap.fromList [(a, ())] compare


structure MatchInfo where
  focus: String
  kinds: RBMap String String compare
  ops: List String -- list to maintain order of introduction. Earliest first
  opArgs: RBMap String (RBMap Nat String compare) compare

def MatchInfo.empty : MatchInfo :=
  { focus := "", kinds := RBMap.empty, ops := [], opArgs := RBMap.empty }

def MatchInfo.replaceOpArg (m: MatchInfo)
  (op: String) -- name of operation
  (oldarg: String) -- name of argument
  (newarg: String): Option MatchInfo :=
  match m.opArgs.find? op with
  | some ix2arg =>
      match ix2arg.toList.find? (fun (ix, name) => name == newarg) with
       | some (ix, _) =>
          let ix2arg' := ix2arg.insert ix newarg
          let opArgs' := m.opArgs.insert op ix2arg'
          some ({ m with opArgs := opArgs'})
       | none => none
  | none => none


inductive Either {a: Type} {b: Type}: Type :=
| Left: a -> Either
| Right: b -> Either

open Either


-- | TODO; write this as a morphism from the torsor over MatchInfo into PDL.
def matcherToMatchInfo (m: matcher) (prev: MatchInfo): MatchInfo :=
match m with
| (matcher.root m) =>
    let cur := matcherToMatchInfo m { focus := "root" , kinds := prev.kinds, ops :=   prev.ops ++ ["root"], opArgs := prev.opArgs }
    matcherToMatchInfo m cur
| (matcher.focus! name m) =>
    let cur :=
      if (prev.ops.contains name)
      then { prev with focus := name }
      else { prev with focus := name, ops := name::prev.ops }
    matcherToMatchInfo m cur
| (matcher.kind? kind m) =>
    let cur := { prev with kinds := prev.kinds.insert (prev.focus) (kind) }
    matcherToMatchInfo m cur
| (matcher.arg? ix name m) =>
    let args := match prev.opArgs.find? prev.focus with
      | none => RBMap.fromList [(ix.toNat, name)] compare
      | some args => args.insert ix.toNat (name)
    let cur := { prev with opArgs := prev.opArgs.insert prev.focus args }
    matcherToMatchInfo m cur
 | (matcher.built) =>  prev


-- | %ty = pdl.type
-- | // right operand of add.
-- | %add_right_rand = pdl.operand : %ty
--
-- //       %0 = "pdl.type"() : () -> !pdl.type
-- //       %1 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
--
-- | an op operand that is not defined by an operation needs a pdl.operand
def freePDLOpOperand (operandName: String) (ix: Int) (parent: SSAVal): List (BasicBlockStmt builtin) :=
    -- | TODO: cleanup, somehow auomatically generate the correct type?
  -- (AttrDict.mk [AttrEntry.mk "index" (AttrVal.int ix (MLIRTy.int 32))])
    let attr := [mlir_attr_dict| { "index" = 42 } ] -- [escape| AttrVal.int ix (MLIRTy.int 32) ] }]
    let rhs := Op.empty "pdl.operand"
    -- let rhs := rhs.addArg parent [mlir_type| !"pdl.operation"]  -- [parent] [] [] attr [mlir_type| ()]
    let rhs := rhs.addResult [mlir_type| !"pdl.value"]
    let lhs := match parent with
      | SSAVal.SSAVal parentName => (SSAVal.SSAVal $ operandName)
    [BasicBlockStmt.StmtAssign lhs none rhs]

def boundPDLOpOperand (operandName: String) (ix: Int) (parent: SSAVal): List (BasicBlockStmt builtin) :=
    -- let attr := (AttrDict.mk [AttrEntry.mk "index" (AttrVal.int ix (MLIRTy.int 32))])
    let rhs := Op.empty "pdl.result"
    let rhs := rhs.addAttr "index" (AttrValue.int ix .i32)
    -- let rhs := rhs.addArg parent [mlir_type| !"pdl.operation"] -- lol what? the parent info is broken
    let rhs := rhs.addArg (SSAVal.SSAVal $ operandName) [mlir_type| !"pdl.operation"]
    let rhs := rhs.addResult [mlir_type| !"pdl.value"]
    let lhs := SSAVal.SSAVal $ operandName ++ "_result"
    [BasicBlockStmt.StmtAssign lhs none rhs]


def opOperandsToPDL (ops: List String) (args: RBMap Nat String compare) (parent: SSAVal): List (BasicBlockStmt builtin) :=
  let args := args.toList
  args.bind (fun ixname =>
    let ix := ixname.fst
    let name := ixname.snd
    if ops.contains name
    then boundPDLOpOperand ixname.snd ixname.fst parent
    else freePDLOpOperand ixname.snd ixname.fst parent)


def opToPDL (m: MatchInfo) (parentName: String): List (BasicBlockStmt builtin) :=
  let args? :=  m.opArgs.find? parentName
  let kind? := m.kinds.find? parentName -- where is this supposed to be used?
  let lhs := SSAVal.SSAVal parentName
  let args : List (BasicBlockStmt builtin) :=
    match args? with
    | some args => (opOperandsToPDL m.ops args (SSAVal.SSAVal parentName))
    | none => []
  let argSSAVals := args.bind
    (fun arg => match arg with
          | BasicBlockStmt.StmtAssign lhs _ _ => [lhs]
          | BasicBlockStmt.StmtOp _ => [])
  let op := Op.empty "pdl.operation"
  let op := match kind? with
    | some kind => op.addAttr "name" kind
    | none => op
  let op := argSSAVals.foldl (fun o a => o.addArg a [mlir_type| !"pdl.value"]) op
  let op := op.addResult [mlir_type| !"pdl.operation"]
  let op := op.addAttr "operand_segment_sizes" (builtin.denseVectorOfList [args.length, 0, 0])
  let op := op.addAttr "attributeNames" (AttrValue.list [])
  args ++ [BasicBlockStmt.StmtAssign lhs none op]

def MatchInfo.toPDL (m: MatchInfo): Op builtin :=
 -- let stmts := m.ops.reverse.map (opToPDL m)
 let stmts := m.ops.map (opToPDL m)
 let rgn := Region.mk [BasicBlock.mk "entry" [] stmts.join]
 [mlir_op| "pdl.pattern" () ($(rgn)) { "benefit" = 1 : i16 } : () -> ()  ]


-- | TODO: monadify this.
partial def matcherToMatchInfoStx (m: Lean.Syntax)
  : Lean.PrettyPrinter.UnexpandM  (@Either Lean.Syntax MatchInfo) :=
match m with
| `(matcher.root $m) =>
      return Right
              ({ focus := "root"
                , kinds := RBMap.empty
                , ops :=  ["root"]
                , opArgs := RBMap.empty
                })
| `(matcher.focus! $name $m) => do
    let eprev <- matcherToMatchInfoStx m
    match eprev with
    | Left stx => return (Left stx)
    | Right prev =>
      match name.raw.isStrLit? with
      | some n => -- update focus
        if (prev.ops.contains n)
        then return Right ({ prev with focus := n })
        else  -- | add new op
          return Right ({ prev with focus := n, ops := n::prev.ops })
      | _ =>  return (Left m)
| `(matcher.kind? $kind $m) => do
    let eprev <- matcherToMatchInfoStx m
    match eprev with
    | Left stx => return (Left stx)
    | Right prev =>
      match kind.raw.isStrLit? with
      | some k => -- | update kind
        return Right ({ prev with kinds := prev.kinds.insert (prev.focus) k })
      | _ =>
          return Left m
          -- return Right ({ prev with kinds := prev.kinds.insert (prev.focus) "UNK_KIND" })
| `(matcher.arg? $ix $name $m) => do
    let eprev <- matcherToMatchInfoStx m
    match eprev with
    | Left stx => return (Left stx)
    | Right prev =>
      match (ix.raw.isNatLit?, name.raw.isStrLit?) with
      | (some ix, some name) =>
        let args := match prev.opArgs.find? prev.focus with
          | none => RBMap.fromList [(ix, name)] compare
          | some args => args.insert ix name
        return Right { prev with opArgs := prev.opArgs.insert prev.focus args }
      | _ => return (Left m) -- prev
| `(matcher.erase! $m) => do
    let eprev <- matcherToMatchInfoStx m
    match eprev with
    | Left stx => return (Left stx)
    | Right prev =>
        let newFocus := "???"
        let newOps := prev.ops.filter (fun op => op != prev.focus)
        return Right { prev with ops := newOps, focus := newFocus }
| `(matcher.replaceOperand! $oldrand $newrand $m) => do
    let eprev <- matcherToMatchInfoStx m
    match eprev with
    | Left stx => return (Left stx)
    | Right prev =>
        match (oldrand.raw.isStrLit?, newrand.raw.isStrLit?) with
        | (some oldrand, some newrand) =>
            match prev.replaceOpArg (prev.focus) oldrand newrand with
            | some matchinfo => return Right matchinfo
            | none => return Left m
        | _ => return Left m

| m => return Left m
-- | m => return Right { focus := toString "UNKNOWN"
--                       , kinds := RBMap.empty
--                       , ops := ["UNKNOWN"]
--                       , opArgs := AssocList.empty }


declare_syntax_cat str_newline

partial def stx_vgroup_strings (ss: Array String)
: Lean.PrettyPrinter.UnexpandM Lean.Syntax := do

  -- let newline :=   Lean.Syntax.atom si "atom"

  let newline :=
    -- Lean.mkNode `antiquot #[Lean.Syntax.atom si "atom"]
    Lean.mkNode Lean.nullKind #[Lean.mkAtom "atom"]
  -- let newline :=
  --   Lean.Syntax.ident si
  --     "\n\n".toSubstring
  --     (Lean.Name.str Lean.Name.anonymous "asd" 1) []
  let mut out <- `("")
  for s in ss do
    out :=  (<- `($out
                  $(⟨newline⟩)
                   $(Lean.quote s)))
  return out

partial def unexpandMatch (m: Lean.Syntax) : Lean.PrettyPrinter.UnexpandM Lean.Syntax := do
  let ematchinfo <- matcherToMatchInfoStx m
  match ematchinfo with
  | Left err => return err
  | Right matchinfo =>
    let mut prettyOps : Array String := #[]
    for opName in matchinfo.ops do
      let mut s := "\n"
      if matchinfo.focus == opName
      then s := ">"
      else s := " "

      s := s ++ " %" ++ opName ++ ":= ";
      match matchinfo.kinds.find? opName with
      | some kind => s := s ++  kind ++ "["
      | none => s := s ++ "??? ["

      -- HACK
      match matchinfo.opArgs.find? opName with
      | none => s := s -- TODO: what the fuck? why do I need this over ()
      | some args =>
        for (argix, argname) in args do
          s := s ++ "(arg" ++ toString argix ++ "=%" ++ argname ++ ")" ++ " "
      s := s ++ "]"
    --   prettyOps := prettyOps.push s
    -- `($(Lean.quote prettyOps))
    let mut outstr : String := "-----\n"
    for s in prettyOps do
      outstr := outstr ++ s ++ "\n"
    return Lean.mkIdent (Lean.Name.append Lean.Name.anonymous outstr)

-- NOTE: piecemeal delaboration
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- Cannot delaborate piecemeal. The problem with this is that
-- if we have (kind (root)),
-- the delaborator for kind on input (kind root) recieves the DELABORATED
-- version of root.
-- We need the delaborator of kind on input (kind root) should recieve
-- the whole thing delaborated.

-- @[appUnexpander matcher.built]
-- partial def unexpandMatcherbuilt : Lean.PrettyPrinter.Unexpander
--   := unexpandMatch

-- @[appUnexpander matcher.kind?]
-- partial def unexpandMatcherKind? : Lean.PrettyPrinter.Unexpander
--   := unexpandMatch

-- @[appUnexpander matcher.arg?]
-- partial def unexpandMatcherArg? : Lean.PrettyPrinter.Unexpander
--   := unexpandMatch

-- @[appUnexpander matcher.focus!]
-- partial def unexpandMatcherFocus! : Lean.PrettyPrinter.Unexpander
--   := unexpandMatch

-- @[appUnexpander matcher.root]
-- partial def unexpandMatcherRoot : Lean.PrettyPrinter.Unexpander
--   := unexpandMatch



-- Can't use this for proofs, because the goal state is a type!
-- So we need to create a goal state where we can pull out the matcher.



-- def proof : matcher := by {
--   apply matcher.root _;
--   apply matcher.kind? "get" _;
--   apply matcher.arg? 1 "y" _;
--   apply matcher.built;
-- }

-- #print proof


inductive built' : matcher -> Type where
| built': built' matcher.built
| root_built': (m: matcher)
      -> (PRF: built' m)
      ->  built' (matcher.root m)
| kind?_built':
  (s: String)
  -> (m: matcher)
  -> (PRF: built' m)
  -> built' (matcher.kind? s m)
| arg?_built':
  (ix: Int)
  -> (s: String)
  -> (m: matcher)
  -> (PRF: built' m)
  -> built' (matcher.arg? ix s m)
| focus!_built': (s: String)
  -> (PRF: built' m)
  -> built' (matcher.focus! s m)

@[appUnexpander built']
partial def unexpandMatcherbuilt'Prop :
Lean.PrettyPrinter.Unexpander :=  fun m =>
match m with
| `(built' $arg) => do
      unexpandMatch arg
| unk => `("built_UNK" $(⟨unk⟩))


-- | does not work, because exists is existential. need sigma type
-- def proof_built_m : matcher := extractMatcher (proof_built)
-- #print proof_built_m


def root' (m: matcher)
  (PRF: built' (matcher.root m) × matcher): built' m × matcher :=
  match PRF with
  -- | TODO: try using n
  | (built'.root_built' m' prf, n) => (prf, (matcher.root n))

def kind?' (s: String)
  (m: matcher)
  (PRF: built' (matcher.kind? s m) × matcher): built' m × matcher :=
  -- | TODO: try using n
  match PRF with
  | (built'.kind?_built' s m prf, n) => (prf, matcher.kind? s n)

def arg?' (ix: Int) (s: String)
  (m: matcher)
  (PRF: built' (matcher.arg? ix s m) × matcher): built' m × matcher :=
  match PRF with
  | (built'.arg?_built' ix s m prf, n) => (prf, matcher.arg? ix s n)

def focus!' (s: String)
  (m: matcher)
  (PRF: built' (matcher.focus! s m) ×  matcher): built' m × matcher :=
  match PRF with
  | (built'.focus!_built' s prf, n) => (prf, matcher.focus! s n)





structure RewriteInfo where
  matchInfo: MatchInfo
  -- | name -> (kind, args)
  -- ops: RBMap String (String × (RBMap Nat String compare)) compare
  replacements: RBMap String String compare

def RewriteInfo.empty : RewriteInfo :=
  { matchInfo := MatchInfo.empty, replacements := RBMap.empty }


inductive rewriter where
-- | create: (name: String) -> (kind: String) -> (args: List String) -> rewriter -> rewriter
| built: rewriter
| replace: (op: String) -> (val: String) -> rewriter -> rewriter
| root: matcher -> rewriter -> rewriter

inductive rewriter_built : rewriter -> Type where
| built: rewriter_built rewriter.built
| root: (m: matcher)
      -> (r: rewriter)
      -> (PRF: rewriter_built r)
      ->  rewriter_built (rewriter.root m r)
| replace: (op: String)
   -> (val: String)
   -> (r: rewriter)
   -> (PRF: rewriter_built r)
   ->  rewriter_built (rewriter.replace op val r)
--   -> (kind: String)
--   -> (args: List String)
--   -> (r: rewriter)
--   -> (PRF: rewriter_built  r)
--   -> rewriter_built (rewriter.create name kind args r)


-- | functions for backward chaining
def rewriter_root (m: matcher) (r: rewriter)
  (PRF: rewriter_built (rewriter.root m r) × rewriter): rewriter_built r × rewriter :=
  match PRF with
  | (rewriter_built.root m _ prf, s) => (prf, (rewriter.root m s))

def rewriter_replace (op: String) (val: String) (r: rewriter)
  (PRF: rewriter_built (rewriter.replace op val r) × rewriter): rewriter_built r × rewriter :=
  match PRF with
  | (rewriter_built.replace op val _ prf, s) => (prf, rewriter.replace op val s)

def rewriter_built' (r: rewriter) (PRF: rewriter_built r):
   rewriter_built rewriter.built × rewriter :=
   (rewriter_built.built, rewriter.built)


-- def rewriter_create (name: String) (kind: String) (args: List String)
--   (r: rewriter)
--   (PRF: rewriter_built (rewriter.create name kind args r) × rewriter): rewriter_built r × rewriter :=
--   match PRF with
--   | (rewriter_built.create name kind args r prf, s) => (prf, rewriter.create name kind args s)


def matcher0tactic : Σ  (m: matcher), (built' m) × matcher := by {
  apply Sigma.mk;
  apply root';
  apply focus!' "zero";
  apply kind?' "asm.zero";
  apply focus!' "root";
  apply kind?' "asm.add";
  apply arg?' 0 "zero";
  apply arg?' 1 "v";
  repeat constructor;
}


-- === MATCHER DEMO ==


def matcher0: matcher := matcher0tactic.snd.snd
#print matcher0


def matcher0pdl: Op builtin := (matcherToMatchInfo matcher0tactic.snd.snd MatchInfo.empty).toPDL
#eval IO.eprintln $ Pretty.doc $ matcher0pdl

-- === REWRITER ==

--  | create a rewriter that starts from matcher0,
-- and builds a rewrite interactively
def rewriter0tactic : Σ  (r: rewriter), (rewriter_built r) × rewriter := by {
  apply Sigma.mk;
  apply rewriter_root matcher0;
  apply rewriter_replace "root" "v";
  repeat constructor;
}



def rewriter0: rewriter := rewriter0tactic.snd.snd
#reduce rewriter0

-- | TODO; write this as a morphism from the torsor over MatchInfo into PDL.
def rewriterToRewriteInfo (ast: rewriter) (state: RewriteInfo): RewriteInfo :=
  match ast with
  | rewriter.root matcher ast =>
    let matchInfo := matcherToMatchInfo matcher MatchInfo.empty
    rewriterToRewriteInfo ast { state with matchInfo := matchInfo }
  | rewriter.built => state
  | rewriter.replace root v ast =>
      let state := { state with replacements := state.replacements.insert root v }
      rewriterToRewriteInfo ast state

def RewriteInfo.toPDL (state: RewriteInfo): Op builtin :=
   let ops := state.replacements.toList.map (fun rootAndReplacement =>
      let root := SSAVal.SSAVal rootAndReplacement.fst
      let replacement := SSAVal.SSAVal rootAndReplacement.snd
      let op : Op builtin := Op.empty "pdl.replace"
      let op := op.addArg root [mlir_type| !"pdl.operation"]
      let op := op.addArg replacement [mlir_type| !"pdl.value"]
      let op := op.addAttr "operand_segment_sizes" (builtin.denseVectorOfList [1, 0, 1])
      BasicBlockStmt.StmtOp op
   )
   let root := SSAVal.SSAVal state.matchInfo.ops.reverse.head! -- jank!
   let rewrite := Op.empty (δ := builtin) "pdl.rewrite"
   let rewrite := rewrite.addArg root [mlir_type| !"pdl.operation"]
   let bb := (BasicBlock.empty "entry").appendStmts ops
   let bb := bb.appendStmt (Op.empty (δ := builtin) "pdl.rewrite_end")
   let rewrite := rewrite.appendRegion bb

   -- apend rewrite to matchop
   let matchop := state.matchInfo.toPDL
   let matchBB := matchop.singletonRegion.singletonBlock
   let matchBB := matchBB.appendStmt rewrite
   let matchop := matchop.mutateSingletonRegion (fun _ => matchBB)
   matchop


def rewriter0pdl: Op builtin := (rewriterToRewriteInfo rewriter0 RewriteInfo.empty).toPDL
#eval IO.eprintln $ Pretty.doc $ rewriter0pdl

def full0pdl : Op builtin := [mlir_op|
   module  {
         $(rewriter0pdl)
   }]
#eval IO.eprintln $ Pretty.doc $ full0pdl

-- | returns success/failure and the new op
def runPattern (rewrite: Op builtin) (code: Op builtin): IO (Option (Op builtin)) := do
  let filepath := "temp.mlir"
  let combinedModule :=
  [mlir_op| module {
    $(rewrite)
    $(code)
  }]
  let outstr := "// RUN: mlir-opt %s  -allow-unregistered-dialect -test-pdl-bytecode-pass \n" ++
      (Pretty.doc combinedModule)
  IO.FS.writeFile filepath outstr
  let args := #["-allow-unregistered-dialect"
               , "-test-pdl-bytecode-pass"
               , "--mlir-print-op-generic"
               , filepath]
  let new_mod_str <- IO.Process.run { cmd := "mlir-opt", args := args }
  let notes := []
  let (loc, notes, _, res) := (pop (δ := builtin) ()).runP locbegin notes new_mod_str
  IO.eprintln (vgroup $ notes.map (note_add_file_content new_mod_str))
  match res with
   | Result.ok op => do
    return op
   | Result.err err => do
      IO.eprintln "***Parse Error:***"
      IO.eprintln (note_add_file_content new_mod_str err)
      return Option.none
   | Result.debugfail err =>  do
      IO.eprintln "***Debug Error:***"
      IO.eprintln (note_add_file_content new_mod_str err)
      return Option.none

-- auto evaluation
def code0 : Op builtin := [mlir_op| module {
  func @"main"() {
    %x = "asm.int" () { "val" = 32 } : () -> (i32)
  }
}]


--#eval runPattern rewriter0pdl code0 >>= IO.println

unsafe def unsafePerformIOImpl [Inhabited a] (io: IO a): a :=
  match unsafeIO io with
  | Except.ok a    =>  a
  | Except.error e => panic! "expected io computation to never fail"

@[implementedBy unsafePerformIOImpl]
def unsafePerformIO [Inhabited a] (io: IO a): a := Inhabited.default

-- | returns the rewritten module.
def unsafeRunRewriterAndExtractModule (rewrite: Op builtin) (code: Op builtin):
    Option (Op builtin) :=
  let out : Option (Op builtin) := unsafePerformIO (runPattern rewrite code)
  match out with
  | none => none
  | some out =>  panic! "TODO: use lens to extract out module"

-- def code1 := runRewriter rewriter0pdl code0
