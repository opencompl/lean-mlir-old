import MLIR.AST
import Lean.PrettyPrinter
import Std.Data.AssocList
import Lean.Meta
import MLIR.EDSL
import MLIR.Doc
open MLIR.Doc
open Lean.PrettyPrinter
open Lean.Meta
open Std
open Std.AssocList
open MLIR.AST
open MLIR.EDSL

-- PDL: Pattern Description Language 
-- |
-- +-> parse it into MLIR data structure --(lean code that spits out a matcher)
-- |
-- +-> matcher

inductive matcher where
| built: matcher
| kind?: (kind: String) -> matcher -> matcher
| arg?: (ix: Int) -> (name: String) -> matcher -> matcher
| focus!: (name: String) -> matcher -> matcher
| root: matcher -> matcher
| erase!: matcher -> matcher
-- ^ erase the currently focused op.
| replaceOp!: (new: String) ->  matcher -> matcher 
-- ^ replace focused op with new op.
| replaceOperand!:  (old: String) -> (new: String) ->  matcher -> matcher 
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
  opArgs: AssocList String (RBMap Nat String compare)

def MatchInfo.empty : MatchInfo := 
  { focus := "", kinds := RBMap.empty, ops := [], opArgs := AssocList.empty }

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
    let cur := matcherToMatchInfo m { focus := "root" , kinds := prev.kinds , ops :=   prev.ops ++ ["root"], opArgs := prev.opArgs }
    matcherToMatchInfo m cur
| (matcher.focus! name m) =>
    let cur := 
      if (prev.ops.contains name) 
      then { prev with focus := name }
      else  -- | add new op
            { prev with focus := name, ops := name::prev.ops }
    matcherToMatchInfo m cur
| (matcher.kind? kind m) => 
    let cur := { prev with kinds := prev.kinds.insert (prev.focus) kind }
    matcherToMatchInfo m cur
          -- return Right ({ prev with kinds := prev.kinds.insert (prev.focus) "UNK_KIND" })
| (matcher.arg? ix name m) =>
    let args := match prev.opArgs.find? prev.focus with
      | none => RBMap.fromList [(ix.toNat, name)] compare
      | some args => args.insert ix.toNat name
    let cur := { prev with opArgs := prev.opArgs.insert prev.focus args }
    matcherToMatchInfo m cur
 | (matcher.built) =>  prev
 | m =>  prev




def opOperandsToPDL (args: RBMap Nat String compare) (parent: SSAVal): List BasicBlockStmt :=
  let args := args.toList
  args.map (fun ixname =>
    let ix := ixname.fst
    let name := ixname.snd
    let attr := (AttrDict.mk [AttrEntry.mk "index" (AttrVal.int ix (MLIRTy.int 32))])
    let rhs := Op.mk "pdl.operand" [parent] [] [] attr [mlir_type| ()]
    let lhs := (SSAVal.SSAVal name)
    (BasicBlockStmt.StmtAssign lhs rhs))

def opToPDL (m: MatchInfo) (parentName: String): List BasicBlockStmt := 
let args? :=  m.opArgs.find? parentName
let kind? := m.kinds.find? parentName 
let lhs := SSAVal.SSAVal parentName
let (op, args) : Op × List BasicBlockStmt := 
  match (args?, kind?) with
          | (some args, some kind) => 
              let op := (Op.mk "pdl.operation" [] [] [] (AttrDict.mk [AttrEntry.mk "kind" (AttrVal.str kind)]) [mlir_type| () ] ) 
              let args := (opOperandsToPDL args (SSAVal.SSAVal parentName))
              (op, args)
              
          | (some args, none) =>  
              let op := (Op.mk "pdl.operation" [] [] [] (AttrDict.mk []) [mlir_type| () ] )
              let args := (opOperandsToPDL args (SSAVal.SSAVal parentName))
              (op, args)
          | (none, some kind) => 
              let op := (Op.mk "pdl.operation" [] [] [] (AttrDict.mk [AttrEntry.mk "kind" (AttrVal.str kind)]) [mlir_type| () ] )
              let args := []
              (op, args)
          | (none, none) =>  
            let op := (Op.mk "pdl.operation" [] [] [] (AttrDict.mk []) [mlir_type| () ] )
            let args := []
            (op, args)
 [BasicBlockStmt.StmtAssign lhs op] ++ args


def matchInfoToPDL (m: MatchInfo): Op :=
 let stmts := m.ops.reverse.map (opToPDL m)
 let rgn := Region.mk [BasicBlock.mk "entry" [] stmts.join]
 [mlir_op| "pdl.pattern" () ([escape| rgn]) : () -> ()  ]


                      


-- | TODO: monadify this.
partial def matcherToMatchInfoStx (m: Lean.Syntax) 
  : Lean.PrettyPrinter.UnexpandM  (@Either Lean.Syntax MatchInfo) :=
match m with
| `(matcher.root $m) => 
      return Right 
              ({ focus := "root"
                , kinds := RBMap.empty
                , ops :=  ["root"]
                , opArgs := AssocList.empty
                })
| `(matcher.focus! $name $m) => do
    let eprev <- matcherToMatchInfoStx m
    match eprev with
    | Left stx => return (Left stx)
    | Right prev => 
      match name.isStrLit? with
      | some n => -- update focus
        if (prev.ops.contains n) 
        then return Right ({ prev with focus := n })
        else  -- | add new op
          return Right ({ prev with focus := n, ops := n::prev.ops })
      | _ =>  (Left m)
| `(matcher.kind? $kind $m) => do
    let eprev <- matcherToMatchInfoStx m
    match eprev with
    | Left stx => return (Left stx)
    | Right prev => 
      match kind.isStrLit? with
      | some k => -- | update kind
        return Right ({ prev with kinds := prev.kinds.insert (prev.focus) k })
      | _ => 
          return Left m 
          -- return Right ({ prev with kinds := prev.kinds.insert (prev.focus) "UNK_KIND" })
| `(matcher.arg? $ix $name $m) => do
    let eprev <- matcherToMatchInfoStx m
    match eprev with
    | Left stx => Left stx
    | Right prev => 
      match (ix.isNatLit?, name.isStrLit?) with
      | (some ix, some name) => 
        let args := match prev.opArgs.find? prev.focus with
          | none => RBMap.fromList [(ix, name)] compare
          | some args => args.insert ix name
        return Right { prev with opArgs := prev.opArgs.insert prev.focus args }
      | _ => Left m -- prev 
| `(matcher.erase! $m) => do 
    let eprev <- matcherToMatchInfoStx m
    match eprev with
    | Left stx => Left stx
    | Right prev => 
        let newFocus := "???"
        let newOps := prev.ops.filter (fun op => op != prev.focus)
        return Right { prev with ops := newOps, focus := newFocus }
| `(matcher.replaceOperand! $oldrand $newrand $m) => do 
    let eprev <- matcherToMatchInfoStx m
    match eprev with
    | Left stx => Left stx
    | Right prev => 
        match (oldrand.isStrLit?, newrand.isStrLit?) with
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

  let si := Lean.SourceInfo.original 
      "".toSubstring 0
      "asd".toSubstring 1
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
                  $newline
                   $(Lean.quote s)))
  return out
  
partial def unexpandMatch (m: Lean.Syntax) : Lean.PrettyPrinter.UnexpandM Lean.Syntax := do
  let ematchinfo <- matcherToMatchInfoStx m
  match ematchinfo with
  | Left err => err
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
      
      match matchinfo.opArgs.find? opName with
      | none => ()
      | some args =>
        for (argix, argname) in args do 
          s := s ++ "(arg" ++ toString argix ++ "=%" ++ argname ++ ")" ++ " "
      s := s ++ "]"
      prettyOps := prettyOps.push s
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
| unk => `("built_UNK" $unk)


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

-- def root' (m: matcher)
--   (PRF: built' (matcher.root m)): 
--   built' m :=
--     match PRF with 
--     | built'.root_built _ prf => prf

def begin' (m: matcher)
  (PRF: built' (matcher.focus! "root" (matcher.root m))): 
  built' m :=
  match PRF with
  | built'.focus!_built' _ prf => 
    match prf with 
    | built'.root_built' _ prf => prf


def root'' (m: matcher) 
  (PRF: built' (matcher.root m)): matcher :=
  match PRF with
  | built'.root_built' _ prf => m



def matcher0tactic : Σ  (m: matcher), (built' m) × matcher := by {
  apply Sigma.mk;
  apply root';
  apply kind?' "get";
  apply arg?' 0 "x2";
  apply arg?' 1 "k";
  apply focus!' "x2";
  apply kind?' "set";
  apply arg?' 0 "x1";
  apply arg?' 1 "k";
  apply focus!' "root"; 
  apply arg?' 3 "bar";
  apply focus!' "x2";
  apply arg?' 2 "v2";
  repeat constructor;
}



def matcher0: matcher :=matcher0tactic.snd.snd
#print matcher0


def matcher0pdl: Op := matchInfoToPDL $ matcherToMatchInfo matcher0tactic.snd.snd MatchInfo.empty
#eval IO.eprintln $ Pretty.doc $  matcher0pdl

