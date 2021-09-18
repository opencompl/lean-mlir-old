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
| mk: (name: String) -> (ops: List Op) -> BasicBlock

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

def P (a: Type) : Type := Loc × String -> Result ParseError (Loc × String × a)

-- https://github.com/leanprover/lean4/blob/d0996fb9450dc37230adea9d10ecfdf10330ef67/tests/playground/flat_parser.lean
def ppure {a: Type} (v: a): P a := λ(loc, s) => Result.ok (loc, s, v)

def pbind {a b: Type} (pa: P a) (a2pb : a -> P b): P b := 
   λs => match pa s with 
            | Result.ok (l, s', a) => a2pb a (l, s')
            | Result.err e => Result.err e

def pfail : P a := λ(loc, s) => 
  Result.err ({ left := loc, right := loc, kind := ErrKind.mk "fail"})


-- | take string upto character delimiter, consuming character delimiter
def pStrUpto (d: Char) : P String := λ(startloc, haystack) => 
  let go (loc: Loc) (s: String) (out: String): Result ParseError (Loc × String × String) :=
      if isEmpty s 
      then Result.err ({left := startloc, right := loc, kind := ErrKind.mk ("expected delimiter but ran out of string")})
      else 
        let c := front haystack;
        if c == d
        then Result.ok (loc, s, out)
        else Result.ok (loc, s, out)  -- go loc (s.drop) (out.extend c)  -- recursion, how?
  go startloc haystack  ""

-- | parse a string and return the a
def pstring (needle: String) (v: a): P a := λ(loc, haystack) =>
 if needle.isPrefixOf haystack
 -- | TODO: how do I write this as loc.advance needle?
 -- | TODO: produce better error of mismatch
 then Result.ok (advance loc needle, drop haystack (length needle), v)
 else Result.err ({left := loc, right := loc, kind := ErrKind.mk ("expected |" ++ needle ++ "|")})

def pchar (needle: Char) (v: a): P a := λ(loc, haystack) =>
  if isEmpty haystack
  then Result.err ({left := loc, right := loc, kind := ErrKind.mk ("expected char |" ++ needle.toString ++ "| found empty")})
  else if front haystack == needle 
  then Result.ok (advanceone loc needle, drop haystack 1, v)
 else Result.err ({left := loc, right := loc, kind := ErrKind.mk ("expected |" ++ needle.toString ++ "| found invalid")})


def pstring_ (s: String) : P Unit := pstring s ()
def pchar_ (c: Char) : P Unit := pchar c ()

-- | parse things intercalated by character c upto character d
def pintercalatedUpto (p: P a) (i: Char) (d: Char) : P (List a):= pfail

-- | parse things intercalated by character c upto character d
def manyUpto (p: P a) (d: Char) : P (List a):= pfail

-- | never fails.
def ppeek (c: Char) : P Bool := λ(loc, haystack) =>
  if isEmpty haystack
  then Result.ok (loc, haystack, False)
  else Result.ok  (loc, haystack, front haystack == 'c')


def pregion : P Region :=  pfail

def poperand : P SSAVal := pfail

-- | "name" (arg1, ..., argn) [(rgn1, ... rgnn)]?
def pop : P Op := do 
  let _ <- pchar_ '"'
  let name <- pStrUpto '\"'
  let args <- pintercalatedUpto poperand ',' ')'
  -- let hasRegion <- ppeek '('
  let hasRegion := true
  let regions <- (if hasRegion 
                   then manyUpto pregion ')' 
                   else ppure [])
  return (Result.ok (Op.mk name args [] regions))


-- def pblock : P BasicBlock := do
--   let c <- pchar_ '^' 
--   let name <- pStrUpto ':'
--   let ops <- pmany ops

instance : Monad P := {
  pure := ppure,
  bind := pbind
}

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


