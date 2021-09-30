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


-- PRETTYPRINTING
-- ===============

inductive Doc : Type where
  | Concat : Doc -> Doc -> Doc
  | Nest : Doc -> Doc
  | VGroup : List Doc -> Doc
  | Text: String -> Doc

instance : Inhabited Doc where
  default := Doc.Text ""


instance : Coe String Doc where
  coe := Doc.Text

instance : Append Doc where 
  append := Doc.Concat

def doc_concat (ds: List Doc): Doc := ds.foldl Doc.Concat (Doc.Text "") 


partial def layout 
  (d: Doc)
  (indent: Int) -- indent
  (width: Int) -- width
  (leftover: Int) -- characters left
  (newline: Bool) -- create newline?
  : String :=
  match d with
    | (Doc.Text s)  => (if newline then "\n".pushn ' ' indent.toNat else "") ++ s
    | (Doc.Concat d1 d2) =>
         let s := layout d1 indent width leftover newline
         s ++ layout d2 indent width (leftover - (length s + 1)) false
    | (Doc.Nest d) => layout d (indent+1) width leftover newline
    | (Doc.VGroup ds) => 
       let ssInline := layout (doc_concat ds) indent width leftover newline 
       if false then ssInline -- length ssInline <= leftover then ssInline
       else  
         let width' := width - indent
         -- TODO: don't make 
         String.join (ds.map (fun d => layout d indent width width True))


def layout80col (d: Doc) : String := layout d 0 80 0 false

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




mutual
partial def op_to_doc (op: Op): Doc := 
    match op with
    | (Op.mk name args attrs rgns) => name ++ " (" ++ Doc.Nest (Doc.VGroup (rgns.map rgn_to_doc) ++ ")")

partial def bb_to_doc(bb: BasicBlock): Doc :=
  match bb with
  | (BasicBlock.mk name args ops) => Doc.VGroup [Doc.Text (name ++ ": "), Doc.Nest (Doc.VGroup (ops.map op_to_doc))]

partial def rgn_to_doc(rgn: Region): Doc :=
  match rgn with
  | (Region.mk bbs) => Doc.VGroup ["{", Doc.Nest (Doc.VGroup (bbs.map bb_to_doc)), "}"]
 
end

instance : ToString Op := {
  toString := fun op =>  layout80col (op_to_doc op)
}


instance : ToString BasicBlock := {
  toString := fun bb => layout80col (bb_to_doc bb)
}

instance : ToString Region := {
  toString := fun rgn => layout80col (rgn_to_doc rgn)
}


-- PARSER
-- ==========



inductive Result (e : Type) (a : Type) : Type where 
| ok: a -> Result e a
| err: e -> Result e a

instance [Inhabited e] : Inhabited (Result e a) where
   default := Result.err (Inhabited.default) 


inductive ErrKind : Type where
| mk : (name : String) -> ErrKind

instance : ToString ErrKind := {
  toString := fun k => 
    match k with
    | ErrKind.mk s => s
}


instance : Inhabited ErrKind where
   default := ErrKind.mk ""


structure Loc where
  line : Int
  column : Int

instance : Inhabited Loc where
   default := { line := 1, column := 1 }


instance : ToString Loc := {
  toString := fun loc => 
    toString loc.line ++ ":" ++ toString loc.column
}


def locbegin : Loc := { line := 1, column := 1 }

-- | move a loc by a string.
def advance (l: Loc) (s: String): Loc :=
  if isEmpty s then l
  else if front s == '\n'
    then { line := l.line + 1, column := 1  }
    else { line := l.line, column := l.column + 1}
 
def advance1 (l: Loc) (c: Char): Loc :=
  if c == '\n'
    then { line := l.line + 1, column := 1  }
    else return { line := l.line, column := l.column + 1}

structure ParseError where
  left : Loc
  right : Loc
  kind : ErrKind


instance : Inhabited ParseError where
   default := { left := Inhabited.default, right := Inhabited.default, kind := Inhabited.default }

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


def perror (err: String) :  P a := {
  runP := λ loc _ =>
     Result.err ({ left := loc, right := loc, kind := ErrKind.mk err})
}

instance : Inhabited (P a) where
   default := perror "INHABITED INSTANCE OF PARSER"

def psuccess (v: a): P a := { 
    runP := λ loc s  => 
      Result.ok (loc, s, v)
  }


-- | eat till '\n'
partial def eat_line_ (l: Loc) (s: String): Loc × String :=
  if isEmpty s then (l, s)
  else let c := front s
  if c == '\n'
  then (l, s)
  else return eat_line_ (advance1 l c) (s.drop 1)

partial def eat_whitespace_ (l: Loc) (s: String) : Loc × String :=
    if isEmpty s
    then (l, s)
    else  
     let c:= front s
     if isPrefixOf "//" s
     then 
      let (l, s) := eat_line_ l s
      eat_whitespace_ l s
     else if c == ' ' || c == '\t'  || c == '\n'
       then eat_whitespace_ (advance1 l c) (s.drop 1)
       else (l, s)


-- | never fails.
def ppeek : P (Option Char) := { 
  runP := λ loc haystack =>
    if isEmpty haystack
    then Result.ok (loc, haystack, none)
    else do
     let (loc, haystack) := eat_whitespace_ loc haystack
     Result.ok (loc, haystack, some (front haystack))
  }

def padvance_char_INTERNAL (c: Char) : P Unit := {
  runP := λ loc haystack => Result.ok (advance1 loc c, drop haystack 1, ())
}

def pconsume(c: Char) : P Unit := do
  let cm <- ppeek
  match cm with 
  | some c => padvance_char_INTERNAL c
  | _ =>  perror ("expected character |" ++ toString c ++ "|. Found: |" ++ toString cm ++ "|." )


def ppeek?(c: Char) : P Bool := do
  let cm <- ppeek
  return (cm == some c)


-- | TOO: convert to token based.
def ppeekPred (pred: Char -> Bool) (default: Bool) : P Bool := do
 let cm <- ppeek
 match cm with
 | some c => return (pred c)
 | none => return default



def eat_whitespace : P Unit := {
  runP := λ loc s =>
    let (l', s') := eat_whitespace_ loc s
    Result.ok (l', s', ())
  }

 


partial def takeWhile (predicate: Char -> Bool)
   (startloc: Loc)
   (loc: Loc)
   (s: String)
   (out: String): Result ParseError (Loc × String × String) :=
      if isEmpty s 
      then Result.err {left := startloc, right := loc, kind := ErrKind.mk ("expected delimiter but ran out of string")}
      else 
        let c := front s;
        if predicate c
        then takeWhile predicate startloc (advance1 loc c) (s.drop 1) (out.push c)
        else Result.ok (loc, s, out)

partial def ptakewhile (predicateWhile: Char -> Bool) : P String :=
{ runP := λ startloc haystack =>  takeWhile predicateWhile startloc startloc haystack ""
}

-- | take an identifier. TODO: ban symbols
def pident : P String := do
  eat_whitespace 
  ptakewhile (fun c => c != ' ' && c != '\t' && c != '\n' && c != ':')

-- | pstar p delim is either (i) a `delim` or (ii) a  `p` followed by (pmany p delim)
partial def pstarUntil (p: P a) (d: Char) : P (List a) := do
   eat_whitespace
   if (<- ppeek? d)
   then do 
     pconsume d
     return []
   else do
       let a <- p
       let as <- pstarUntil p d
       return (a::as)


-- | pdelimited l p r is an l, followed by as many ps, followed by r.
partial def pdelimited (l: Char) (p: P a) (r: Char) : P (List a) := do
  pconsume l
  pstarUntil p r


-- parse an <r> or a <i> <p> <pintercalated_>
partial def pintercalated_ (p: P a) (i: Char) (r: Char) : P (List a) := do
  match (<- ppeek) with
   | some c => if c == r
               then do pconsume r;return []
               else if c == i
               then do
                 pconsume i
                 let a <- p
                 let as <- pintercalated_ p i r
                 return (a :: as)
               else perror ("expected |" ++ i.toString ++ "|, or |" ++ r.toString ++ "|, found|" ++ c.toString ++ "|.")
   | _ =>  perror ("expected |" ++ i.toString ++ "|, or |" ++ r.toString ++ "|, found EOF" )


-- | parse things intercalated by character c upto character d
partial def pintercalated (l: Char) (p: P a) (i: Char) (r: Char) : P (List a) := do
  pconsume l
  match (<- ppeek) with
   | some c => if c == r
               then do pconsume r; return []
               else do
                  let a <- p
                  let as <- pintercalated_ p i r 
                  return (a :: as)
   | _ => perror "expected either ')' or a term to be parsed. Found EOF"


partial def pstr : P String :=  do
   eat_whitespace
   pconsume '"'
   let s <- ptakewhile (fun c => c != '"')
   pconsume '"'
   return s


-- | ppeekstar peeks for `l`.
-- | (a) If it finds `l`, it returns `p` followed by `ppeekstar l`.
-- |(ii) If it does not find `l`, it retrns []
partial def ppeekstar (l: Char) (p: P a) : P (List a) := do
  let proceed <- ppeek? l
  if proceed then do 
        let a <- p
        let as <- ppeekstar l p
        return (a :: as)
  else return []

mutual
partial def pssaval : P SSAVal := perror "pssaval"

-- | mh, needs to be mutual. Let's see if LEAN lets me do this.
partial def pregion : P Region :=  do
  let rs <- pdelimited '{' pbb '}'
  return (Region.mk rs)


-- | parse <whitespace> "..."
   
partial def poperand : P SSAVal := perror "poperandImpl"

partial def pop : P Op := do 
  eat_whitespace
  match (<- ppeek) with 
  | some '\"' => do
    let name <- pstr
    let args <- pintercalated '(' poperand ',' ')'
    let hasRegion <- ppeek? '('
    let regions <- (if hasRegion 
                      then pdelimited '(' pregion ')' 
                      else ppure [])
     return (Op.mk  name args [] regions)
  | some '%' => perror "found %, don't know how to parse ops yet"
  | other => perror ("expected '\"' or '%' to begin operation definition. found: " ++ toString other)




partial def popbinding : P (SSAVal × Op) := do
   let val <- pssaval
   pconsume '='
   let op <- pop
   return (val, op)
   


partial def pblockImpl : P BasicBlock := do
   pconsume '^'
   let name <- pstr -- actually should be identifier?
   let args <- pintercalated '(' poperand ',' ')'
   pconsume ':'
   let ops <- ppeekstar '%' pop
   return (BasicBlock.mk name args ops)
end  



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
  let res := pop.runP locbegin contents
  match res with
   | Result.ok (loc, str, op) => IO.println op
   | Result.err res => IO.println res
  return ()
