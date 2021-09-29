import Init.Data.String
import Init.Data.String.Basic
import Init.System.IO
import Lean.Parser
import Lean.Parser.Extra
import Init.System.Platform
import Init.Data.String.Basic
import Init.Data.Repr
import Init.Data.ToString.Basic


-- /home/bollu/work/lean4/tests/lean/server
-- import Lean.Data.Lsp
-- open IO Lean Lsp


open String
open Lean
open Lean.Parser
open IO
open System


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

-- | TODO: create lambda case.
instance : ToString Op := {
  toString := fun op => 
    match op with
    | (Op.mk name args attrs region) => name
}


-- PARSER
-- ==========

inductive Result (e : Type) (a : Type) : Type where 
| ok: a -> Result e a
| err: e -> Result e a

inductive ErrKind : Type where
| mk : (name : String) -> ErrKind

instance : ToString ErrKind := {
  toString := fun k => 
    match k with
    | ErrKind.mk s => s
}

structure Loc where
  line : Int
  column : Int

instance : ToString Loc := {
  toString := fun loc => 
    toString loc.line ++ ":" ++ toString loc.column
}


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


instance : ToString ParseError := {
  toString := fun err => 
    toString err.left ++ " " ++ toString err.kind
}


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
      Result.err ({ left := loc, right := loc, kind := ErrKind.mk "pfail"})
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

-- | todo: move loc
def pconsume (c: Char) : P Unit := do
  let b <- ppeek c
  if b then psuccess () else pfail

def pident (s: String) : P Unit := { 
  runP := λ loc haystack =>
    if haystack.take (s.length) == s
    then Result.ok (advance loc s, drop haystack (s.length), ())
    else Result.err { left := loc, right := loc, kind := ErrKind.mk ("expected identifier |" ++ s ++ "|") }
  }



-- | take string upto character delimiter, consuming character delimiter
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


-- workaround: https://github.com/leanprover/lean4/issues/697
constant pssaval : P SSAVal
constant pregion : P Region
constant pstr : P String
constant poperand : P SSAVal
constant pop : P Op
constant popbinding : P (SSAVal × Op)
constant ppeekstar (l: Char) (p: P a) : P (List a)
constant pblock : P BasicBlock

mutual

partial def pssavalImpl : P SSAVal := pfail


-- | mh, needs to be mutual. Let's see if LEAN lets me do this.
partial def pregionImpl : P Region :=  do
  let rs <- pdelimited '{' pblock '}'
  return (Region.mk rs)


-- | parse "..."
partial def pstrImpl : P String := pfail
partial def poperandImpl : P SSAVal := pfail

partial def popImpl : P Op := do 
  let name <- pstr
  let args <- pintercalated '(' poperand ',' ')'
  let hasRegion <- ppeek '('
  let regions <- (if hasRegion 
                   then do 
                     pdelimited '(' pregion ')' 
                   else ppure [])
  return (Op.mk name args [] regions)


partial def popbindingImpl : P (SSAVal × Op) := do
   let val <- pssaval
   pconsume '='
   let op <- pop
   return (val, op)
   


-- | ppeekstar peeks for `l`.
-- | (a) If it finds `l`, it returns `p` followed by `ppeekstar l`.
-- |(ii) If it does not find `l`, it retrns []
partial def ppeekstarImpl (l: Char) (p: P a) : P (List a) := do
  let proceed <- ppeek l
  if proceed then do 
        let a <- p
        let as <- ppeekstar l p
        return (a :: as)
  else return []

partial def pblockImpl : P BasicBlock := do
   pconsume '^'
   let name <- pstr -- actually should be identifier?
   let args <- pintercalated '(' poperand ',' ')'
   pconsume ':'
   let ops <- ppeekstar '%' pop
   return (BasicBlock.mk name args ops)
end  


attribute [implementedBy pssavalImpl] pssaval
attribute [implementedBy pregionImpl] pregion
attribute [implementedBy pstrImpl] pstr
attribute [implementedBy poperandImpl] poperand
attribute [implementedBy popImpl] pop
attribute [implementedBy popbindingImpl] popbinding
attribute [implementedBy ppeekstarImpl] ppeekstar
attribute [implementedBy pblockImpl] pblock


partial def ptoplevel : P Op := do
  let _ <- pident "module"
  -- pconsume '{'
  -- pconsume '}'
  return (Op.mk "module" [] [] [])


-- TOPLEVEL PARSER
-- ==============

-- https://github.com/leanprover/lean4/blob/master/tests/playground/file.lean
def main (xs: List String): IO Unit := do
  -- let path : System.FilePath :=  xs.head!
  let path :=  xs.head!
  let contents ← FS.readFile path;
  IO.println "FILE\n====\n"
  IO.println contents
  IO.println "PARSING\n=======\n"
  let res := ptoplevel.runP locbegin contents
  match res with
   | Result.ok (loc, str, op) => IO.println op
   | Result.err res => IO.println res
  return ()
