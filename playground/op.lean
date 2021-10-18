import Lean.PrettyPrinter
import Std.Data.AssocList
import Lean.Meta
open Lean.PrettyPrinter
open Lean.Meta
open Std
open Std.AssocList

inductive matcher where
| done: matcher
| kind?: (kind: String) -> matcher -> matcher
| arg?: (ix: Int) -> (name: String) -> matcher -> matcher
| focus!: (name: String) -> matcher -> matcher
| root: matcher -> matcher


abbrev Set (α : Type) (compare: α -> α -> Ordering) := RBMap α Unit compare

instance {α: Type} {β: Type} {compare: α -> α -> Ordering} : Inhabited (RBMap α β compare) where
  default := RBMap.empty

def set_union {α: Type} {compare:α -> α -> Ordering} (xs: Set α compare) (ys: Set α compare): Set α compare := 
  RBMap.fromList (xs.toList ++ ys.toList) compare

def set_insert{α: Type} {compare:α -> α -> Ordering} 
  (xs: Set α compare) (a: α): Set α compare :=
  RBMap.insert xs a ()


def RBMap.set_bind {α: Type} {compare: α -> α -> Ordering} (xs: Set α compare) (f: α -> Set α compare): Set α compare :=
  RBMap.fold (fun new a () => set_union new (f a)) RBMap.empty xs 

def set_map {α: Type} {compare: α -> α -> Ordering} (xs: Set α compare) (f: α -> β) (compare': β -> β -> Ordering): Set β compare' := 
    RBMap.fold (fun new a () => RBMap.insert new (f a) ()) RBMap.empty xs

def set_subtract {α: Type} {compare: α -> α -> Ordering} (all: Set α compare) (to_remove: Set α compare): Set α compare :=
  RBMap.fold (fun all' a () => RBMap.erase all' a) to_remove all 

def set_singleton {α: Type} (a: α) (compare: α -> α -> Ordering): Set α compare :=
    RBMap.fromList [(a, ())] compare


structure MatchInfo where
  focus: String
  kinds: RBMap String String compare
  ops: List String -- list to maintain order of introduction. Earliest first
  opArgs: AssocList String (RBMap Nat String compare)

partial def computeMatcher_ (m: Lean.Syntax) 
  : Lean.PrettyPrinter.UnexpandM  MatchInfo :=
match m with
| `(matcher.root $m) => 
      return  { focus := "root"
                , kinds := RBMap.empty
                , ops :=  ["root"]
                , opArgs := AssocList.empty
                }
| `(matcher.focus! $name $m) => do
    let prev <- computeMatcher_ m
    match name.isStrLit? with
    | some n => -- update focus
      if (prev.ops.contains n) 
      then return { prev with focus := n }
      else  -- | add new op
        return { prev with focus := n, ops := n::prev.ops }
    | _ =>  prev
| `(matcher.kind? $kind $m) => do
    let prev <- computeMatcher_ m
    match kind.isStrLit? with
    | some k => -- | update kind
      return { prev with kinds := prev.kinds.insert (prev.focus) k }
    | _ =>  prev
| `(matcher.arg? $ix $name $m) => do
    let prev <- computeMatcher_ m
    match (ix.isNatLit?, name.isStrLit?) with
    | (some ix, some name) => 
      let args := match prev.opArgs.find? prev.focus with
        | none => RBMap.fromList [(ix, name)] compare
        | some args => args.insert ix name
      return { prev with opArgs := prev.opArgs.insert prev.focus args }
    | _ => prev 
| m => return { focus := toString ""
                , kinds := RBMap.empty
                , ops := []
                , opArgs := AssocList.empty }


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
  let matchinfo <- computeMatcher_ m
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


inductive matcher_done : matcher -> Prop where
| done: matcher_done matcher.done
| root_done: (m: matcher) 
      -> (PRF: matcher_done m)
      ->  matcher_done (matcher.root m)
| kind?_done: 
  (s: String)
  -> (m: matcher)
  -> (PRF: matcher_done m)
  -> matcher_done (matcher.kind? s m)
| arg?_done: 
  (ix: Int)
  -> (s: String)
  -> (m: matcher)
  -> (PRF: matcher_done m)
  -> matcher_done (matcher.arg? ix s m)
| focus!_done: (s: String)
  -> (PRF: matcher_done m)
  -> matcher_done (matcher.focus! s m)


@[appUnexpander matcher_done]
partial def unexpandMatcherDoneProp : 
Lean.PrettyPrinter.Unexpander :=  fun m => 
match m with
| `(matcher_done $arg) => do
      unexpandMatch arg
| unk => `("MATCHER_DONE_UNK" $unk)


-- def root (m: matcher) 
--   (PRF: matcher_done (matcher.root m)): matcher_done m :=
--   match PRF with
--   | matcher_done.root _ prf => prf

def kind? (s: String)
  (m: matcher)
  (PRF: matcher_done (matcher.kind? s m)): matcher_done m :=
  match PRF with
  | matcher_done.kind?_done s m prf => prf

def arg? (ix: Int) (s: String)
  (m: matcher)
  (PRF: matcher_done (matcher.arg? ix s m)): matcher_done m :=
  match PRF with
  | matcher_done.arg?_done ix s m prf => prf

def focus! (s: String)
  (m: matcher)
  (PRF: matcher_done (matcher.focus! s m)): matcher_done m :=
  match PRF with
  | matcher_done.focus!_done s prf => prf

def root (m: matcher)
  (PRF: matcher_done (matcher.root m)): 
  matcher_done m :=
    match PRF with 
    | matcher_done.root_done _ prf => prf

def begin (m: matcher)
  (PRF: matcher_done (matcher.focus! "root" (matcher.root m))): 
  matcher_done m :=
  match PRF with
  | matcher_done.focus!_done _ prf => 
    match prf with 
    | matcher_done.root_done _ prf => prf

-- %x2 = set %x1 %k %v
-- %root = get %x2 %k
def proof : ∃ m, matcher_done m := by {
  apply Exists.intro;
  apply root;
  apply kind? "get";
  apply arg? 0 "x2";
  apply arg? 1 "k";
  apply focus! "x2";
  apply kind? "set";
  apply arg? 0 "x1";
  apply arg? 1 "k";
  apply arg? 2 "v";
  apply focus! "root"; 

  repeat constructor;
}


-- inductive op where
-- | add: String -> op -> op
-- | done: op

-- inductive is_done : op -> Prop where
--   | is_done_done: is_done op.done
--   | is_done_add: (s: String) -> (x: op) -> is_done x -> is_done (op.add s x)

-- -- | do case analysis on PRF, show that it must have been created by is_done_add.
-- -- TODO: how do I use `inversion`?
-- def is_done_step_bwd (s: String) (o: op) (PRF: is_done (op.add s o)): is_done o  := 
-- match PRF with | is_done.is_done_add _ _ prf => prf


-- def is_done_step (s: String) (o: op) (PRF: is_done o): is_done (op.add s o) := 
--   is_done.is_done_add s o PRF
-- -- | build stepwise.
-- -- TODO: this lives in prop, so you cannot eliminate into Type.
-- -- fuck large elimination x(


-- def proof : ∃ o, is_done o := by {
--   apply Exists.intro;
--   apply is_done_step_bwd "foo";
--   apply is_done_step_bwd "bar";
--   apply is_done.is_done_add;
--   apply is_done.is_done_add;
--   apply is_done.is_done_done;
-- }



  
    
   
-- def main: IO Unit :=
--   IO.println "foo"



