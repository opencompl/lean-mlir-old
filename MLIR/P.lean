-- parser combinator library
import MLIR.AST
import MLIR.Doc

open MLIR.AST
open MLIR.Doc
open Pretty
open String
open Char

namespace MLIR.P

inductive Result (e : Type) (a : Type) : Type where
| ok: a -> Result e a
| err: e -> Result e a
| debugfail : e -> Result e a

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
  ix : Int

instance : Inhabited Loc where
   default := { line := 1, column := 1, ix := 0 }

instance : Pretty Loc where
   doc (loc: Loc) := toString loc.line ++ ":" ++ toString loc.column


def locbegin : Loc := { line := 1, column := 1, ix := 0 }


def advance1 (l: Loc) (c: Char): Loc :=
  if c == '\n'
    then { line := l.line + 1, column := 1, ix := l.ix + 1  }
    else { line := l.line, column := l.column + 1, ix := l.ix + 1}

-- | move a loc by a string.
partial def advance (l: Loc) (s: String): Loc :=
  if isEmpty s then l
  else let c := s.front; advance (advance1 l c) (s.drop 1)


structure Note where
  left : Loc
  right : Loc
  kind : Doc


instance : Inhabited Note where
   default :=
     { left := Inhabited.default
       , right := Inhabited.default
       , kind := Inhabited.default }

instance : Pretty Note where
  doc (note: Note) :=
      doc note.left ++ " " ++  note.kind


-- | TODO: enable notes, refactor type into Loc x String x [Note] x (Result ParseError a)
structure P (a: Type) where
   runP: Loc -> List Note -> String ->  (Loc × (List Note) × String × (Result Note a))



-- | map for parsers
def pmap (f : a -> b) (pa: P a): P b := {
  runP :=  λ loc ns s =>
    match pa.runP loc ns s with
      | (l, ns, s, Result.ok a) => (l, ns,  s, Result.ok (f a))
      | (l, ns, s, Result.err e) => (l, ns, s, Result.err e)
      | (l, ns, s, Result.debugfail e) => (l, ns, s, Result.debugfail e)
}


-- https://github.com/leanprover/lean4/blob/d0996fb9450dc37230adea9d10ecfdf10330ef67/tests/playground/flat_parser.lean
def ppure {a: Type} (v: a): P a := { runP :=  λ loc ns s =>  (loc, ns, s, Result.ok v) }

def pbind {a b: Type} (pa: P a) (a2pb : a -> P b): P b :=
   { runP := λloc ns s => match pa.runP loc ns s with
            | (l, ns, s, Result.ok a) => (a2pb a).runP l ns  s
            | (l, ns, s, Result.err e) => (l, ns, s, Result.err e)
            | (l, ns, s, Result.debugfail e) => (l, ns, s, Result.debugfail e)
   }

instance : Monad P := {
  pure := ppure,
  bind := pbind
}


def pnote [Pretty α] (a: α): P Unit := {
  runP := λ loc ns s =>
    let n := { left := loc, right := loc, kind := doc a }
    (loc, ns ++ [n], s, Result.ok ())
}

def perror [Pretty e] (err: e) :  P a := {
  runP := λ loc ns s =>
     (loc, ns, s, Result.err ({ left := loc, right := loc, kind := doc err}))
}

def pdebugfail [Pretty e] (err: e) :  P a := {
  runP := λ loc ns s =>
     (loc, ns, s, Result.debugfail ({ left := loc, right := loc, kind := doc err}))
}


instance : Inhabited (P a) where
   default := perror "INHABITED INSTANCE OF PARSER"

def psuccess (v: a): P a := {
    runP := λ loc ns s  =>
      (loc, ns, s, Result.ok v)
  }


def pmay (p: P a): P (Option a) := {
    runP := λ loc ns s  =>
      match p.runP loc ns s with
        |  (loc, ns, s, Result.ok v) => (loc, ns, s, Result.ok (Option.some v))
        | (loc, ns, s, Result.err e) => (loc, ns, s, Result.ok Option.none)
        | (l, ns, s, Result.debugfail e) => (l, ns, s, Result.debugfail e)
  }


-- try p. if success, return value. if not, run q
-- TODO: think about what to do about notes from p in por.
def por (p: P a) (q: P a) : P a :=  {
  runP := λ loc ns s =>
    match p.runP loc ns s with
      | (loc', ns', s', Result.ok a) => (loc', ns', s', Result.ok a)
      | (loc', ns', s', Result.err e) => q.runP loc ns s
      | (l, ns, s, Result.debugfail e) => (l, ns, s, Result.debugfail e)
}

-- def pors (ps: List (p a)) : P a :=
--  match ps with
--  | [] => []
--  | [p] => p
--  | (p::ps) por p (pors ps)


-- | eat till '\n'
partial def eat_line_ (l: Loc) (s: String): Loc × String :=
  if isEmpty s then (l, s)
  else let c := front s
  if c == '\n'
  then (l, s)
  else eat_line_ (advance1 l c) (s.drop 1)

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
  runP := λ loc ns haystack =>
    if isEmpty haystack
    then (loc, ns, haystack, Result.ok none)
    else
     let (loc, haystack) := eat_whitespace_ loc haystack
     (loc, ns, haystack, Result.ok ∘ some ∘ front $ haystack)
  }

def padvance_char_INTERNAL (c: Char) : P Unit := {
  runP := λ loc ns haystack => (advance1 loc c, ns, drop haystack 1, Result.ok ())
}

def pconsume(c: Char) : P Unit := do
  let cm <- ppeek
  match cm with
  | some c' =>
     if c == c' then padvance_char_INTERNAL c
     else perror ("pconsume: expected character |" ++ toString c ++ "|. Found |" ++ toString c' ++ "|.")
  | none =>  perror ("pconsume: expected character |" ++ toString c ++ "|. Found EOF")


def ppeek?(c: Char) : P Bool := do
  let cm <- ppeek
  return (cm == some c)


def eat_whitespace : P Unit := {
  runP := λ loc ns s =>
    let (l', s') := eat_whitespace_ loc s
    (l', ns, s', Result.ok ())
  }


partial def takeWhile (predicate: Char -> Bool)
   (startloc: Loc)
   (loc: Loc)
   (s: String)
   (out: String):  (Loc × String × Result Note String) :=
      if isEmpty s
      then (loc, s, Result.err {left := startloc,
                                right := loc,
                                kind := "expected delimiter but ran out of string"})
      else
        let c := front s;
        if predicate c
        then takeWhile predicate startloc (advance1 loc c) (s.drop 1) (out.push c)
        else (loc, s, Result.ok out)

partial def ptakewhile (predicateWhile: Char -> Bool) : P String :=
{ runP := λ startloc ns haystack =>
      let (loc, s) := takeWhile predicateWhile startloc startloc haystack ""
      (loc, ns, s)
}



-- | take an identifier. TODO: ban symbols
def pident : P String := do
  eat_whitespace
  ptakewhile (fun c => (c != ' ' && c != '\t' && c != '\n') && (isAlphanum c || c == '_'))

def pident? (s: String) : P Unit := do
   let i <- pident
   pnote $ "pident? looking for ident: " ++ s ++ " | found: |" ++ i ++ "|"
   if i == s
   then psuccess ()
   else perror $ "expected |" ++ s ++ "| but found |" ++ i ++ "|"


def pnat : P Nat := do
  eat_whitespace
  let name <- ptakewhile (fun c => c.isDigit)
  match name.toNat? with
   | some num => return num
   | none => perror $ "expected natural, found |" ++ name ++ "|."

def pnumber : P Int := do
  eat_whitespace
  let name <- ptakewhile (fun c => c.isDigit)
  match name.toInt? with
   | some num => return num
   | none => perror $ "expected number, found |" ++ name ++ "|."

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


-- parse an [ <r> | <i> <p> <pintercalated_> ]
partial def pintercalated_ (p: P a) (i: Char) (r: Char) : P (List a) := do
  eat_whitespace
  match (<- ppeek) with
   | some c => -- perror ("intercalate: I see |" ++ c.toString ++ "|")
               if c == r
               then do pconsume r; return []
               else if c == i
               then do
                 pconsume i
                 eat_whitespace
                 let a <- p
                 let as <- pintercalated_ p i r
                 return (a :: as)
               else perror ("intercalate: expected |" ++ doc i ++ "|  or |" ++ doc r ++ "|, found |" ++ c.toString ++ "|.")
   | _ =>  perror ("intecalate: expected |" ++ doc i ++ "|  or |" ++ doc r ++ "|, found EOF" )


-- | parse things starting with a <l>, followed by <p> intercalated by <i>, ending with <r>
partial def pintercalated (l: Char) (p: P a) (i: Char) (r: Char) : P (List a) := do
  eat_whitespace
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


partial def  pmany0 [Pretty a] (p: P a) : P (List a) := do
  match (<- pmay p) with
    | Option.some a => do
        -- pnote $ "pmany0: found " ++ doc a
        let as <- pmany0 p
        return (a::as)
    | Option.none =>
        -- pnote $ "pmany0: found none"
       return []

-- | parse <p>+ for a given <p>
partial def  pmany1 [Pretty a] (p: P a) : P (List a) := do
  let a1 <- p
  let as <- pmany0 p
  return (a1::as)



-- | find minimum k such that
-- s[pos + k*dir] == '\n'
partial def find_newline_in_dir
   (s: String)
   (pos: Int)
   (dir: Int): Int :=
 if pos <= 0
 then 0
 else if pos >= s.length -1  then s.length - 1
 else if s.get (Pos.mk pos.toNat) == '\n' then pos - dir
 else find_newline_in_dir s (pos + dir) dir


-- | find rightmost newline in s[0, pos].
partial def find_earlier_newline
   (s: String)
   (pos: Int): Int := find_newline_in_dir s pos (-1)

-- | find leftmost newline in s[pos, end]
partial def find_later_newline
   (s: String)
   (pos: Int): Int := find_newline_in_dir s pos 1


-- | add a pointer showing file contents at the line where `note` lies.
def note_add_file_content (contents: String) (note: Note): Doc :=
  let ixl := find_earlier_newline contents note.left.ix
  let ixr := find_later_newline contents note.right.ix
  -- | closed interval
  let len := ixr - ixl + 1
  let substr : Substring := (contents.toSubstring.drop ixl.toNat).take len.toNat
  let nspaces : Int := note.left.ix - ixl
  let underline : String :=   ("".pushn ' ' nspaces.toNat).push '^'
  vgroup [doc "---|" ++ note.kind ++ "|---"
          , doc note.left ++ " " ++ substr.toString
          , doc note.left ++ " " ++ underline
          , doc "---"]
