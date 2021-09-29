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
    else Result.ok  (loc, haystack, front haystack == c)
  }


partial def eat_whitespace_ (l: Loc) (s: String) : Loc × String :=
    if isEmpty s
    then (l, s)
    else  
     let c:= front s
     if c == ' ' || c == '\t'  || c == '\n'
     then eat_whitespace_ (advance1 l c) (s.drop 1)
     else (l, s)

def eat_whitespace : P Unit := {
  runP := λ loc s =>
    let (l', s') := eat_whitespace_ loc s
    Result.ok (l', s', ())
  }

 
-- | Exact match a string
def pexact (s: String) : P Unit := { 
  runP := λ loc haystack =>
    if haystack.take (s.length) == s
    then Result.ok (advance loc s, drop haystack (s.length), ())
    else Result.err { left := loc, right := loc, kind := ErrKind.mk ("expected identifier |" ++ s ++ "|") }
  }

def pconsume (c: Char) : P Unit := pexact c.toString


-- | match preceded by whitespace.
def pWhitespaceExact (s: String) : P Unit := do
  eat_whitespace
  pexact s

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

partial def ptakeWhile (predicateWhile: Char -> Bool) : P String :=
{ runP := λ startloc haystack =>  takeWhile predicateWhile startloc startloc haystack ""
}

-- | take an identifier. TODO: ban symbols
def pident : P String := ptakeWhile (fun c => c != ' ' && c != '\t' && c != '\n' && c != ':')

-- | consume as long as predicate is true
partial def pstarUntilPredicate (p: P a) (continue?: Char -> Bool) : P (List a) := do
   eat_whitespace
   let c <- ppeek
   if continue? c
   then do 
       let a <- p
       let as <- pstarUntil p d
       return (a::as)
  else return []



-- | pstar p delim is either (i) a `delim` or (ii) a  `p` followed by (pmany p delim)
partial def pstarUntil (p: P a) (d: Char) : P (List a) := do
   eat_whitespace
   if (<- ppeek d)
   then do 
     pconsume d
     return []
   else do
       let a <- p
       let as <- pstarUntil p d
       return (a::as)




-- | pdelimited l p r is an l, followed by as many ps, followed by r
partial def pdelimited (l: Char) (p: P a) (r: Char) : P (List a) := do
  pconsume l
  pstarUntil p r



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


-- https://mlir.llvm.org/docs/LangRef/
partial def pOp : P Op := do
  eat_whitespace
  pexact "return"
  return (Op.mk "return" [] [] [])

partial def pBB : P BasicBlock := do
   pexact "^"
   let name <- pident
   pexact ":"
   eat_whitespace
   let ops <- pstarUntil pOp '^' 
   return (BasicBlock.mk name [] [op])

partial def pRegion : P Region := do
  pWhitespaceExact "{"
  let bbs <- pstarUntil pBB '}'
  return (Region.mk bbs)


partial def pfunc : P Op := do
  pWhitespaceExact "func"
  eat_whitespace
  pexact "@"
  let name <- pident
  let body <- pRegion
  return (Op.mk "func" [] [] [body])

partial def pmodule : P Op := do
  let _ <- pWhitespaceExact "module"
  pWhitespaceExact "{"
  let fs <- pstarUntil pfunc '}'
  -- pWhitespaceExact "}"
  return (Op.mk "module" [] [] [Region.mk [BasicBlock.mk "entry" [] fs]])


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
  let res := pmodule.runP locbegin contents
  match res with
   | Result.ok (loc, str, op) => IO.println op
   | Result.err res => IO.println res
  return ()
