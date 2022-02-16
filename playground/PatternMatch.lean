import Lean.PrettyPrinter
import Std.Data.AssocList
open Lean.Meta
open Std
open Std.AssocList

-- http://casperbp.net/posts/2019-04-nondeterminism-using-a-free-monad/index.html
-- | c is commands, r is responses.
inductive Free (C: Type) (R: C -> Type) (A: Type) where
| Return: A -> Free C R A
| Bind: (c: C) -> (R c -> Free C R A) -> Free C R A


-- | w a -> a ~= ((a -> r) -> r) --- cps encode.
inductive Cofree (C: Type) (R: C -> Type) (A: Type) where
| Index:  ((v: Type) -> (c: C) -> (R c -> v) -> v) -> Cofree C R A
-- | torsor 
class Torsor (a: Type) (d: Type) where
  unit: d
  delta: a -> a -> d


-- laws: running a cofree to get a value ~= 

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



def matcher_to_string(m: matcher): String :=
 match m with 
  | matcher.built => "built"
  | matcher.root m => "root, " ++ matcher_to_string m
  | matcher.focus! _ m => "focus, " ++ matcher_to_string m
  | matcher.arg? x y m=> "arg, " ++ matcher_to_string m
  | matcher.kind? k m => "kind, " ++ matcher_to_string m
   


instance : ToString matcher where
  toString m := matcher_to_string m

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


@[appUnexpander matcher]
partial def unexpandMatcherp : Lean.PrettyPrinter.Unexpander :=  fun m =>  unexpandMatch m



@[appUnexpander built']
partial def unexpandMatcherbuilt'Prop : Lean.PrettyPrinter.Unexpander :=  fun m => 
match m with
| `(built' $arg) => do unexpandMatch arg
| unk => `("built_UNK" $unk)


inductive refl : (a: Type k) -> a -> a -> Type (k+1) where
| root: (v: a) -> refl a v  v
| layer: (f: a -> a) -> (r: refl a (f w) v) -> refl a w (f v)



 @[appUnexpander refl]
 partial def unexpandRefl : Lean.PrettyPrinter.Unexpander :=  fun m => 
 match m with
 | `(refl $ty $arg1 $arg2) => do unexpandMatch arg1
 | unk => `("refl_UNK" $unk)



-- | does not work, because exists is existential. need sigma type
-- def proof_built_m : matcher := extractMatcher (proof_built)
-- #print proof_built_m

def apply_to_refl {a: Type} (f: a -> a) (v w: a)  (ra: refl a (f w) v): refl a w (f v) :=
  refl.layer f ra


  
inductive typeval: (t: Type) -> t ->  Type 2 where
| val2type: (x: t) -> typeval t x 


def matcher_by_refl : Σ  (m : matcher), Σ (n: matcher), (refl matcher m n) := by {
 apply Sigma.mk;
 apply Sigma.mk;
 apply apply_to_refl matcher.root;
 apply apply_to_refl (matcher.focus! "foo");
 apply apply_to_refl (matcher.kind? "bar");
 repeat constructor;
 }


def m0 : matcher := matcher_by_refl.fst
def m1 : matcher := matcher_by_refl.snd.fst
#eval IO.eprintln m0
#eval IO.eprintln m1
#print m0


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


