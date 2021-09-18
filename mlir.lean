import Init.Data.String
import Init.Data.String.Basic
import Lean.Parser
import Lean.Parser.Extra
import Init.System.Platform
import Init.Data.String.Basic
import Init.Data.Repr
import Init.Data.ToString.Basic


open String
open Lean
open Lean.Parser

namespace mlir


-- EMBEDDING
-- ==========

mutual
inductive SSAVal : Type where
  | SSAVal : String -> SSAVal

inductive Attribute : Type where
  | Attribute: (key: String) 
      -> (value: String)
      -> Attribute

inductive Op : Type where 
 | mk: (name: String) 
      -> (args: List SSAVal)
      -> (attrs: List Attribute)
      -> (region: List Region) -> Op

inductive Path : Type where 
 | PathComponent: (regionix : Int) 
    -> (bbix: Int) 
    -> (opix: Int)
    -> (rec: Path)
    -> Path
 | Path

inductive BasicBlock: Type where
| mk: (name: String) -> (args: List SSAVal) -> (ops: List Op) -> BasicBlock

inductive Region: Type where
| mk: (bbs: List BasicBlock) -> Region
end


-- PARSER
-- ==========

inductive Result (e : Type) (a : Type) : Type where 
| ok: a -> Result e a
| err: e -> Result e a

inductive ErrKind : Type where
| mk : (name : String) -> ErrKind

structure Loc where
  line : Int
  column : Int

def locbegin : Loc := { line := 1, column := 1 }

-- | move a loc by a string.
-- | TODO: this is incorrect!
def advance (l: Loc) (s: String): Loc :=
  if isEmpty s then l
  else if front s == '\n'
    then { line := l.line + 1, column := 1  }
    else return { line := l.line, column := l.column + 1}
 
def advanceone (l: Loc) (c: Char): Loc :=
  if c == '\n'
    then { line := l.line + 1, column := 1  }
    else return { line := l.line, column := l.column + 1}

structure ParseError where
  left : Loc
  right : Loc
  kind : ErrKind

structure P (a: Type) where 
   runP: Loc -> String -> Result ParseError (Loc × String × a)




-- https://github.com/leanprover/lean4/blob/d0996fb9450dc37230adea9d10ecfdf10330ef67/tests/playground/flat_parser.lean
def ppure {a: Type} (v: a): P a := { runP :=  λ loc s => Result.ok (loc, s, v) }

def pbind {a b: Type} (pa: P a) (a2pb : a -> P b): P b := 
   { runP := λloc s => match pa.runP loc s with 
            | Result.ok (l, s', a) => (a2pb a).runP l  s'
            | Result.err e => Result.err e
   }

instance : Monad P := {
  pure := ppure,
  bind := pbind
}


def pfail : P a := { 
    runP := λ loc _  => 
      Result.err ({ left := loc, right := loc, kind := ErrKind.mk "fail"})
  }

instance : Inhabited (P a) where
   default := pfail

def psuccess (v: a): P a := { 
    runP := λ loc s  => 
      Result.ok (loc, s, v)
  }


-- | never fails.
def ppeek (c: Char) : P Bool := { 
  runP := λ loc haystack =>
    if isEmpty haystack
    then Result.ok (loc, haystack, False)
    else Result.ok  (loc, haystack, front haystack == 'c')
  }

def pconsume (c: Char) : P Unit := do
  let b <- ppeek c
  if b then psuccess () else pfail



-- -- | take string upto character delimiter, consuming character delimiter
def puptoraw (d: Char) : P String :=
{ runP := λ startloc haystack => 
  let go (loc: Loc) (s: String) (out: String): Result ParseError (Loc × String × String) :=
      if isEmpty s 
      then Result.err {left := startloc, right := loc, kind := ErrKind.mk ("expected delimiter but ran out of string")}
      else 
        let c := front haystack;
        if c == d
        then Result.ok (loc, s, out)
        else Result.ok (loc, s, out)  -- go loc (s.drop) (out.extend c)  -- recursion, how?
  go startloc haystack  ""
}

-- | pstar p d is either (i) a `d` or (ii) a  `p` followed by (pmany p d)
partial def pstar (p: P a) (d: Char) : P (List a) := do
   let done <- ppeek d
   if done
   then return []
   else do
       let a <- p
       let as <- pstar p d
       return (a::as)

-- | pdelimited l p r is an l, followed by as many ps, followed by r
partial def pdelimited (l: Char) (p: P a) (r: Char) : P (List a) := do
  pconsume l
  pstar p r


-- | parse things intercalated by character c upto character d
def pintercalated (l: Char) (p: P a) (i: Char) (r: Char) : P (List a) := pfail

mutual
partial def pssaval : P SSAVal := pfail


-- | mh, needs to be mutual. Let's see if LEAN lets me do this.
partial def pregion : P Region :=  do
  let rs <- pdelimited '{' pblock '}'
  return (Region.mk rs)


-- | parse "..."
partial def pstr : P String := pfail
partial def poperand : P SSAVal := pfail

partial def pop : P Op := do 
  let name <- pstr
  let args <- pintercalated '(' poperand ',' ')'
  let hasRegion <- ppeek '('
  let regions <- (if hasRegion 
                   then do 
                     pdelimited '(' pregion ')' 
                   else ppure [])
  return (Op.mk name args [] regions)


partial def popbinding : P (SSAVal × Op) := do
   let val <- pssaval
   pconsume '='
   let op <- pop
   return (val, op)
   


-- | ppeekstar peeks for `l`.
-- | (a) If it finds `l`, it returns `p` followed by `ppeekstar l`.
-- |(ii) If it does not find `l`, it retrns []
partial def ppeekstar (l: Char) (p: P a) : P (List a) := do
  let proceed <- ppeek l
  if proceed then do 
        let a <- p
        let as <- ppeekstar l p
        return (a :: as)
  else return []

partial def pblock : P BasicBlock := do
   pconsume '^'
   let name <- pstr -- actually should be identifier?
   let args <- pintercalated '(' poperand ',' ')'
   pconsume ':'
   let ops <- ppeekstar '%' pop
   return (BasicBlock.mk name args ops)
end  


-- #eval pop.runP locbegin "foo bar"  

-- parseMLIRModule :: String -> Parser Module 

-- EDSL MACRO
-- ==========

notation:100 "{" "}" => Region.Region  []
notation:100 "{" bbs "}" => Region.Region bbs
notation:100 "module" r => Op.mk "module" [] [] [r]
-- notation:100 "^" name ":" => BasicBlock.mk name []
-- https://arxiv.org/pdf/2001.10490.pdf#page=11
macro "^" n:str ":" : term => `(BasicBlock.mk $n [])
macro "^" n:str ":" ops:term : term => `(BasicBlock.BasicBlock $n $ops)


