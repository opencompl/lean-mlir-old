import Init.Data.String
import Init.Data.String.Basic
import Init.Data.Char.Basic
import Init.System.IO
import Lean.Parser
import Lean.Parser.Extra
import Init.System.Platform
import Init.Data.String.Basic
import Init.Data.Repr
import Init.Data.ToString.Basic



-- https://mlir.llvm.org/docs/LangRef/
-- /home/bollu/work/lean4/tests/lean/server
-- import Lean.Data.Lsp
-- open IO Lean Lsp


open String
open Char
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


class Pretty (a : Type) where
  doc : a -> Doc

open Pretty

def vgroup [Pretty a] (as: List a): Doc :=
  Doc.VGroup (as.map doc)

def nest_vgroup [Pretty a] (as: List a): Doc :=
  Doc.Nest (vgroup as)


  


instance : Pretty Doc where
  doc (d: Doc) := d

instance : Pretty String where
  doc := Doc.Text

instance : Pretty Int where
  doc := Doc.Text ∘ toString

instance : Pretty Char where
  doc := Doc.Text ∘ toString

instance : Inhabited Doc where
  default := Doc.Text ""


instance : Coe String Doc where
  coe := Doc.Text

instance : Append Doc where 
  append := Doc.Concat

def doc_dbl_quot : Doc :=  doc '"'

def doc_surround_dbl_quot [Pretty a] (v: a): Doc := 
    doc_dbl_quot ++ doc v ++ doc_dbl_quot
  

def doc_concat (ds: List Doc): Doc := ds.foldl Doc.Concat (Doc.Text "") 

partial def intercalate_doc_rec_ [Pretty d] (ds: List d) (i: Doc): Doc :=
  match ds with
  | [] => Doc.Text ""
  | (d::ds) => i ++ (doc d) ++ intercalate_doc_rec_ ds i

partial def  intercalate_doc [Pretty d] (ds: List d) (i: Doc): Doc := match ds with
 | [] => Doc.Text ""
 | [d] => doc d
 | (d::ds) => (doc d) ++ intercalate_doc_rec_ ds i


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

instance : Coe Doc String where
   coe := layout80col

-- EMBEDDING
-- ==========

inductive BBName
| mk: String -> BBName

instance : Pretty BBName where
  doc name := match name with 
              | BBName.mk s => doc s

mutual
inductive MLIRTy : Type where
| fn : MLIRTy -> MLIRTy -> MLIRTy
| int : Int -> MLIRTy
| tuple : List MLIRTy -> MLIRTy
| vector: Int -> MLIRTy -> MLIRTy

inductive SSAVal : Type where
  | SSAVal : String -> SSAVal

inductive AttrVal : Type where
| str : String -> AttrVal
| int : Int -> MLIRTy -> AttrVal
| type :MLIRTy -> AttrVal
| dense: Int -> MLIRTy -> AttrVal -- dense<10> : vector<i32>

inductive Attr : Type where
  | mk: (key: String) 
      -> (value: AttrVal)
      -> Attr

inductive Op : Type where 
 | mk: (name: String) 
      -> (args: List SSAVal)
      -> (bbs: List BBName)
      -> (regions: List Region) 
      -> (attrs: List Attr)
      -> (ty: MLIRTy)
      -> Op



inductive Path : Type where 
 | PathComponent: (regionix : Int) 
    -> (bbix: Int) 
    -> (opix: Int)
    -> (rec: Path)
    -> Path
 | Path

inductive BasicBlockStmt : Type where
| StmtAssign : SSAVal -> Op -> BasicBlockStmt
| StmtOp : Op -> BasicBlockStmt


inductive BasicBlock: Type where
| mk: (name: String) -> (args: List (SSAVal × MLIRTy)) -> (ops: List BasicBlockStmt) -> BasicBlock




inductive Region: Type where
| mk: (bbs: List BasicBlock) -> Region
end


partial instance :  Pretty MLIRTy where
  doc (ty: MLIRTy) :=
    let rec  go (ty: MLIRTy) :=  
    match ty with
    | MLIRTy.int k => "i" ++ doc k
    | MLIRTy.tuple ts => "(" ++ (intercalate_doc (ts.map go) (doc ", ") ) ++ ")"
    | MLIRTy.fn dom codom => (go dom) ++ " -> " ++ (go codom)
    | MLIRTy.vector sz ty => "vector<" ++ toString sz ++ "x" ++ go ty ++ ">"
    go ty



instance : Pretty AttrVal where
 doc (v: AttrVal) := 
   match v with
   | AttrVal.str str => doc_surround_dbl_quot str 
   | AttrVal.type ty => doc ty
   | AttrVal.int i ty => doc i ++ " : " ++ doc ty
   | AttrVal.dense i ty => "dense<" ++ doc i ++ ">" ++ ":" ++ doc ty


instance : Pretty Attr where
  doc (attr: Attr) := 
    match attr with
    | Attr.mk k v => k ++ " = " ++ (doc v)





instance : Pretty SSAVal where
   doc (val: SSAVal) := 
     match val with
     | SSAVal.SSAVal name => Doc.Text ("%" ++ name)



-- | TODO: add a typeclass `Pretty` for things that can be converted to `Doc`.
mutual
partial def op_to_doc (op: Op): Doc := 
    match op with
    | (Op.mk name args bbs rgns attrs ty) => 
        let doc_name := doc_surround_dbl_quot name 
        let doc_bbs := if bbs.isEmpty
                       then doc ""
                       else "[" ++ intercalate_doc bbs ", " ++ "]"
        let doc_rgns := 
            if rgns.isEmpty
            then Doc.Text ""
            else " (" ++ nest_vgroup (rgns.map rgn_to_doc) ++ ")"
        let doc_args := "(" ++ intercalate_doc args ", " ++ ")"
        let doc_attrs :=
          if List.isEmpty attrs
          then Doc.Text ""
          else "{" ++ intercalate_doc attrs  ", " ++ "}"
        doc_name ++ doc_args ++  doc_rgns ++ doc_attrs ++ " : " ++ doc ty

partial def bb_stmt_to_doc (stmt: BasicBlockStmt): Doc :=
  match stmt with
  | BasicBlockStmt.StmtAssign lhs rhs => (doc lhs) ++ " = " ++ (op_to_doc rhs)
  | BasicBlockStmt.StmtOp rhs => (op_to_doc rhs)

partial def bb_to_doc(bb: BasicBlock): Doc :=
  match bb with
  | (BasicBlock.mk name args stmts) => 
     let doc_arg (arg: SSAVal × MLIRTy) := 
        match arg with
        | (ssaval, ty) => doc ssaval ++ ":" ++ doc ty
     let bbargs := 
        if args.isEmpty then Doc.Text ""
        else "(" ++ 
             (intercalate_doc (args.map doc_arg) ", ") ++ 
             ")"
     let bbname := "^" ++ name ++ bbargs ++ ":"
     let bbbody := Doc.Nest (Doc.VGroup (stmts.map bb_stmt_to_doc))
     Doc.VGroup [bbname, bbbody]

partial def rgn_to_doc(rgn: Region): Doc :=
  match rgn with
  | (Region.mk bbs) => "{" ++ Doc.VGroup [nest_vgroup (bbs.map bb_to_doc), "}"]
 
end

instance : Pretty Op where
  doc := op_to_doc

instance : Pretty BasicBlockStmt where
  doc := bb_stmt_to_doc

instance : Pretty BasicBlock where
  doc := bb_to_doc

instance : Pretty Region where
  doc := rgn_to_doc

instance [Pretty a] : ToString a where
  toString (v: a) := layout80col (doc v)



-- PARSER
-- ==========



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
    else return { line := l.line, column := l.column + 1, ix := l.ix + 1}

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
  runP := λ loc ns haystack =>
    if isEmpty haystack
    then (loc, ns, haystack, Result.ok none)
    else do
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

-- | ^ <name>
def pbbname : P BBName := do 
  pconsume '^'
  let name <- pident
   return (BBName.mk name)

-- | % <name>
partial def pssaval : P SSAVal := do
  eat_whitespace
  pconsume '%'
  let name <- pident
  return (SSAVal.SSAVal name)


mutual

-- | mh, needs to be mutual. Let's see if LEAN lets me do this.
partial def pregion (u: Unit) : P Region :=  do
  pconsume '{'
  -- HACK: entry block need not print block header. See: examples/region-with-no-args.mlir
  let b <- (if (<- ppeek? '^')
           then pblock u 
           else pentryblock_no_label u)

  let bs <- (if (<- ppeek? '^')
            then pmany1 (pblock u)
            else ppure [])
  pconsume '}'
  return (Region.mk (b::bs))



partial def ptype_vector : P MLIRTy := do
  pident? "vector"
  pconsume '<'
  let sz <- pnumber
  pconsume 'x'
  let ty <- ptype ()
  pconsume '>'
  return MLIRTy.vector sz ty
  
partial def ptype (u: Unit) : P MLIRTy := do
  eat_whitespace
  let dom <- (match (<- ppeek) with
             | some '(' => do
                let args <- pintercalated '(' (ptype u) ',' ')'
                return MLIRTy.tuple args
             | some 'i' => do
                 pconsume 'i'
                 let num <- pnumber
                 return MLIRTy.int num
             | other => do
                perror ("uknown type starting with |" ++ toString other ++ "|."))
  eat_whitespace
  match (<- ppeek? '-')  with
  | true => do
        pconsume '-'
        pconsume '>' -- consume arrow
        let codom <- (ptype u)
        return MLIRTy.fn dom codom
  | false => do
     return dom

partial def pblockoperand : P (SSAVal × MLIRTy) := do
  eat_whitespace
  let operand <- pssaval
  pconsume ':'
  let ty <- (ptype ())
  return (operand, ty)


-- | either a string, or a type. Can be others.

partial def pattrvalue_int : P AttrVal := do
  let num <- pnumber
  pconsume ':'
  let ty <- ptype ()
  return AttrVal.int num ty

partial def pattrvalue_dense : P AttrVal := do
  pident? "dense"
  pconsume '<'
  let v <- pnumber
  pconsume '>'
  pconsume ':'
  let ty <- ptype_vector
  return AttrVal.dense v ty
   
  
  
 
partial def pattrvalue : P AttrVal := do
 pnote "hunting for attribute value"
 por pattrvalue_int $ por (pmap AttrVal.str pstr) $ por (pmap AttrVal.type (ptype ())) pattrvalue_dense

partial def pattr : P Attr := do
  eat_whitespace
  let name <- pident
  eat_whitespace
  pconsume '='
  let value <- pattrvalue
  return (Attr.mk name value)

  

partial def pop (u: Unit) : P Op := do 
  eat_whitespace
  match (<- ppeek) with 
  | some '\"' => do
    let name <- pstr
    let args <- pintercalated '(' pssaval ',' ')'
    let bbs <- (if (<- ppeek? '[' ) 
                then pintercalated '[' pbbname ','  ']'
                else return [])
    let hasRegion <- ppeek? '('
    let regions <- (if hasRegion 
                      then pintercalated '(' (pregion ()) ',' ')'
                      else pure [])
    -- | parse attributes
    let hasAttrs <- ppeek? '{'
    let attrs <- (if hasAttrs
              then  pintercalated '{' pattr ',' '}' 
              else pure [])
    pconsume ':'
    let ty <- ptype u
    return (Op.mk  name args bbs regions attrs ty)
  | some '%' => perror "found %, don't know how to parse ops yet"
  | other => perror ("expected '\"' or '%' to begin operation definition. found: " ++ toString other)


partial def popcall (u: Unit) : P BasicBlockStmt := do
   eat_whitespace
   if (<- ppeek? '%')
   then do 
     let val <- pssaval
     pconsume '='
     let op <- pop u
     return (BasicBlockStmt.StmtAssign val op)
   else do
     let op <- pop u
     return (BasicBlockStmt.StmtOp op)

-- | parse a sequence of ops, with no label
partial def pentryblock_no_label (u: Unit) : P BasicBlock := do
   let ops <- pmany1 (popcall u)
   pnote $ "pentry ops: " ++ List.toString ops
   return (BasicBlock.mk "entry" [] ops)


   
partial def pblock (u: Unit) : P BasicBlock := do
   pconsume '^'
   let name <- pident
   let args <- pintercalated '(' pblockoperand ',' ')'
   pconsume ':'
   let ops <- pmany1 (popcall u)
   -- pnote $ "pblock ops: " ++ List.toString ops
   return (BasicBlock.mk name args ops)
end  

-- EDSL
-- ====

declare_syntax_cat mlir_bb
declare_syntax_cat mlir_region
declare_syntax_cat mlir_bb_stmt
declare_syntax_cat mlir_op_results
declare_syntax_cat mlir_op_call
declare_syntax_cat mlir_op_call_args
declare_syntax_cat mlir_op_call_type
declare_syntax_cat mlir_op_operand
declare_syntax_cat mlir_type


-- syntax strLit mlir_op_call_args ":" mlir_op_call_type : mlir_op_call -- no region
-- 


-- EDSL OPERANDS
-- ==============

syntax "%" ident : mlir_op_operand

syntax "mlir_op_operand% " mlir_op_operand : term -- translate operands into term
macro_rules
  | `(mlir_op_operand% % $x:ident) => `(SSAVal.SSAVal $(Lean.quote (toString x.getId))) 

def xx := (mlir_op_operand% %x)
def xxx := (mlir_op_operand% %x)
#print xx
#print xxx


-- EDSL OP-CALL-ARGS
-- =================

syntax "(" ")" : mlir_op_call_args
syntax "(" mlir_op_operand ")" : mlir_op_call_args
syntax "(" mlir_op_operand "," mlir_op_operand","* ")" : mlir_op_call_args

syntax "mlir_op_call_args% " mlir_op_call_args : term -- translate mlir_op_call args into term
macro_rules
  | `(mlir_op_call_args% ( ) ) => `([])
  | `(mlir_op_call_args% ( $x:mlir_op_operand ) ) => `([mlir_op_operand% $x])
  | `(mlir_op_call_args% ( $x:mlir_op_operand, $y:mlir_op_operand ) ) => `([mlir_op_operand% $x, mlir_op_operand% $y])


def call0 : List SSAVal := (mlir_op_call_args% ())
def call1 : List SSAVal := (mlir_op_call_args% (%x))
def call2 : List SSAVal := (mlir_op_call_args% (%x, %y))
#print call0
#print call1
#print call2


-- EDSL MLIR TYPES
-- ===============

syntax "(" ")" : mlir_type
syntax "(" mlir_type ")" : mlir_type
syntax "(" mlir_type "," mlir_type ")" : mlir_type
syntax mlir_type "->" mlir_type : mlir_type
syntax "i"numLit : mlir_type

syntax "mlir_type%" mlir_type : term

macro_rules
  | `(mlir_type% ( ) ) => `(MLIRTy.tuple [])
  | `(mlir_type% ( $x:mlir_type ) ) => `(MLIRTy.tuple [(mlir_type% $x)])
  | `(mlir_type% ( $x:mlir_type, $y:mlir_type ) ) => `(MLIRTy.tuple [(mlir_type% $x), (mlir_type% $y)])
  | `(mlir_type% i $x:numLit ) => `(MLIRTy.int $x)
  | `(mlir_type% $dom:mlir_type -> $codom:mlir_type) => `(MLIRTy.fn (mlir_type% $dom) (mlir_type% $codom))

def ty0 : MLIRTy := (mlir_type% ())
def tyi32 : MLIRTy := (mlir_type% i 32) -- TODO: how to keep no gap?
-- def tyi32' : MLIRTy := (mlir_type% i32) -- TODO: how to keep no gap?
def tysingle : MLIRTy := (mlir_type% (i 42))
def typair : MLIRTy := (mlir_type% (i 32, i 64))
def tyfn0 : MLIRTy := (mlir_type% () -> ())
def tyfn1 : MLIRTy := (mlir_type% (i 11) -> (i 12))
def tyfn2 : MLIRTy := (mlir_type% (i 21, i 22) -> (i 23, i 24))
#print ty0
#print tyi32
#print typair
#print tyfn0
#print tyfn1
-- #print tyi32'


-- EDSL MLIR OP CALL
-- =====================

syntax strLit mlir_op_call_args ":" mlir_type : mlir_op_call

syntax "mlir_op_call%" mlir_op_call : term

macro_rules
  | `(mlir_op_call% $name:strLit $args:mlir_op_call_args : $ty:mlir_type ) =>
        `(Op.mk $name -- name
                (mlir_op_call_args% $args) -- args
                [] -- bbs
                [] -- regions
                [] -- attrs
                (mlir_type% $ty)) -- type


-- | test simple ops [no regions]
def opcall1 : Op := (mlir_op_call% "foo" (%x, %y) : (i 32, i 32) -> i 32)
#print opcall1

-- EDSL MLIR BASIC BLOCK STMT
-- ==========================



syntax mlir_op_call: mlir_bb_stmt
syntax mlir_op_operand "=" mlir_op_call : mlir_bb_stmt
syntax "mlir_bb_stmt%" mlir_bb_stmt : term


macro_rules
  | `(mlir_bb_stmt% $call:mlir_op_call ) =>
       `(BasicBlockStmt.StmtOp (mlir_op_call% $call))
  | `(mlir_bb_stmt% $res:mlir_op_operand = $call:mlir_op_call) => 
       `(BasicBlockStmt.StmtAssign (mlir_op_operand% $res) (mlir_op_call% $call))


def bbstmt1 : BasicBlockStmt := (mlir_bb_stmt% "foo"(%x, %y) : (i 32, i 32) -> i 32)
#print bbstmt1
def bbstmt2: BasicBlockStmt := (mlir_bb_stmt% %z = "foo"(%x, %y) : (i 32, i 32) -> i 32)
#print bbstmt2

-- EDSL MLIR BASIC BLOCK OPERANDS
-- ==============================

declare_syntax_cat mlir_bb_operand
syntax mlir_op_operand ":" mlir_type : mlir_bb_operand

syntax "mlir_bb_operand%" mlir_bb_operand : term

macro_rules 
| `(mlir_bb_operand% $name:mlir_op_operand : $ty:mlir_type ) => 
     `( (mlir_op_operand% $name, mlir_type% $ty) ) 

def bbop1 : SSAVal × MLIRTy := mlir_bb_operand% %x : i 32
#print bbop1



-- EDSL MLIR BASIC BLOCKS
-- ======================


syntax "^" ident ":" (ws mlir_bb_stmt ws)* : mlir_bb
syntax "^" ident "(" sepBy(mlir_bb_operand, ",") ")" ":" (ws mlir_bb_stmt ws)* : mlir_bb

syntax "mlir_bb%" mlir_bb : term

macro_rules 
| `(mlir_bb% ^ $name:ident ( $operands,* ) : $[ $stmts ]* ) => do
   let initList <- `([])
   let argsList <- operands.getElems.foldlM (init := initList) fun xs x => `($xs ++ [mlir_bb_operand% $x])
   let opsList <- stmts.foldlM (init := initList) fun xs x => `($xs ++ [mlir_bb_stmt% $x])
   `(BasicBlock.mk $(Lean.quote (toString name.getId)) $argsList $opsList)
| `(mlir_bb% ^ $name:ident : $[ $stmts ]* ) => do
   let initList <- `([])
   let opsList <- stmts.foldlM (init := initList) fun xs x => `($xs ++ [mlir_bb_stmt% $x])
   `(BasicBlock.mk $(Lean.quote (toString name.getId)) [] $opsList)

def bb1NoArgs : BasicBlock := 
  (mlir_bb%
     ^entry:
     "foo"(%x, %y) : (i 32, i 32) -> i 32
      %z = "bar"(%x) : (i 32) -> (i 32)
      "std.return"(%x0) : (i 42) -> ()

  )
#print bb1NoArgs

def bb2SingleArg : BasicBlock := 
  (mlir_bb%
     ^entry(%argp : i 32):
     "foo"(%x, %y) : (i 32, i 32) -> i 32
      %z = "bar"(%x) : (i 32) -> (i 32)
      "std.return"(%x0) : (i 42) -> ()

  )
#print bb2SingleArg


def bb3MultipleArgs : BasicBlock := 
  (mlir_bb%
     ^entry(%argp : i 32, %argq : i 64):
     "foo"(%x, %y) : (i 32, i 32) -> i 32
      %z = "bar"(%x) : (i 32) -> (i 32)
      "std.return"(%x0) : (i 42) -> ()

  )
#print bb3MultipleArgs


-- EDSL MLIR REGIONS
-- =================

syntax "{" (ws mlir_bb ws)* "}": mlir_region
syntax "mlir_region% " mlir_region : term


macro_rules
| `(mlir_region% { $[ $bbs ]* }) => do
   let initList <- `([])
   let bbsList <- bbs.foldlM (init := initList) fun xs x => `($xs ++ [mlir_bb% $x])
   `(Region.mk $bbsList)

def rgn0 : Region := (mlir_region%  { })
#print rgn0

def rgn1 : Region := 
  (mlir_region%  { 
    ^entry:
      "std.return"(%x0) : (i 42) -> ()
  })
#print rgn1

def rgn2 : Region := 
  (mlir_region%  { 
    ^entry:
      "std.return"(%x0) : (i 42) -> ()

    ^loop:
      "std.return"(%x1) : (i 42) -> ()
  })
#print rgn2

-- MLIR OPS WITH REGIONS
-- =====================

-- Now that we have regions, can extend the grammar to allow ops with regions :D

syntax strLit mlir_op_call_args "(" mlir_region,* ")" ":" mlir_type : mlir_op_call

macro_rules 
  | `(mlir_op_call% $name:strLit $args:mlir_op_call_args ( $rgns,* ) : $ty:mlir_type ) => do
        let initList <- `([])
        let rgnsList <- rgns.getElems.foldlM (init := initList) fun xs x => `($xs ++ [mlir_region% $x])
        `(Op.mk $name -- name
                (mlir_op_call_args% $args) -- args
                [] -- bbs
                $rgnsList -- regions
                [] -- attrs
                (mlir_type% $ty)) -- type

def oprgn0 : Op := (mlir_op_call%
 "func"() ( {
  ^bb0(%arg0: i 32, %arg1: i 32):
    %x = "std.addi"(%arg0, %arg1) : (i 32, i 32) -> i 32
    "std.return"(%x) : (i 32) -> ()
  }) : () -> ()
)
#print oprgn0

-- MLIR ATTRIBUTE VALUE
-- ====================

declare_syntax_cat mlir_attr_val

syntax str: mlir_attr_val
syntax mlir_type : mlir_attr_val

syntax "mlir_attr_val%" mlir_attr_val : term

macro_rules 
  | `(mlir_attr_val% $s:strLit) => `(AttrVal.str $s)
  | `(mlir_attr_val% $ty:mlir_type) => `(AttrVal.type (mlir_type% $ty))


def attrVal0Str : AttrVal := mlir_attr_val% "foo"
#print attrVal0Str

def attrVal1Ty : AttrVal := mlir_attr_val% (i 32, i 64) -> i 32
#print attrVal1Ty

-- MLIR ATTRIBUTE
-- ===============

declare_syntax_cat mlir_attr

syntax ident "=" mlir_attr_val : mlir_attr

syntax "mlir_attr%" mlir_attr : term

macro_rules 
  | `(mlir_attr% $name:ident  = $v:mlir_attr_val) => 
     `(Attr.mk $(Lean.quote (toString name.getId))  (mlir_attr_val% $v))

def attr0Str : Attr := (mlir_attr% sym_name = "add")
#print attr0Str

def attr1Type : Attr := (mlir_attr% type = (i 32, i 32) -> i 32)
#print attr1Type

-- MLIR OPS WITH ATTRIBUTES
-- =====================

-- Now that we have attributes, can extend the grammar to allow ops with regions :D

syntax strLit mlir_op_call_args "{" sepBy(mlir_attr, ",") "}" ":" mlir_type : mlir_op_call

macro_rules 
  | `(mlir_op_call% $name:strLit $args:mlir_op_call_args { $attrs,* } : $ty:mlir_type ) => do
        let initList <- `([])
        let attrsList <- attrs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [mlir_attr% $x])
        `(Op.mk $name -- name
                (mlir_op_call_args% $args) -- args
                [] -- bbs
                [] -- regions   
                $attrsList -- attrs
                (mlir_type% $ty)) -- type

def opattr0 : Op := (mlir_op_call%
 "foo"() { sym_name = "add", type = (i 32, i 32) -> i 32 } : () -> ()
)
#print opattr0



-- MLIR OPS WITH REGIONS AND ATTRIBUTES
-- ====================================


syntax strLit mlir_op_call_args "(" mlir_region,* ")"  "{" mlir_attr,* "}" ":" mlir_type : mlir_op_call

macro_rules 
  | `(mlir_op_call% $name:strLit $args:mlir_op_call_args ( $rgns,* ) { $attrs,* } : $ty:mlir_type ) => do
        let initList <- `([])
        let attrsList <- attrs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [mlir_attr% $x])
        let rgnsList <- rgns.getElems.foldlM (init := initList) fun xs x => `($xs ++ [mlir_region% $x])
        `(Op.mk $name -- name
                (mlir_op_call_args% $args) -- args
                [] -- bbs
                $rgnsList -- regions
                $attrsList -- attrs
                (mlir_type% $ty)) -- type

-- | note that this is a "full stack" example!
def opRgnAttr0 : Op := (mlir_op_call%
 "module"() (
 {
  ^entry:
   "func"() (
    {
     ^bb0(%arg0:i 32, %arg1:i 32):
      %zero = "std.addi"(%arg0 , %arg1) : (i 32, i 32) -> i 32
      "std.return"(%zero) : (i 32) -> ()
    }){sym_name = "add", type = (i 32, i 32) -> i 32} : () -> ()
   "module_terminator"() : () -> ()
 }) : () -> ()
)
#print opRgnAttr0


-- | 
partial def find_newline_in_dir
   (s: String)
   (pos: Int)
   (dir: Int): Int :=
 if pos <= 0
 then 0
 else if pos >= s.length -1  then s.length - 1
 else if s.get pos.toNat == '\n' then pos - dir
 else find_newline_in_dir s (pos + dir) dir


-- | find rightmost newline in s[0, pos].
partial def find_earlier_newline
   (s: String)
   (pos: Int): Int := find_newline_in_dir s pos (-1)

-- | find leftmost newline in s[pos, end]
partial def find_later_newline
   (s: String)
   (pos: Int): Int := find_newline_in_dir s pos 1



-- | add a pointer showing the file contents at the given line
def note_add_file_content (contents: String) (note: Note): Doc :=
  let ixl := find_earlier_newline contents (note.left.ix)
  let ixr := find_later_newline contents (note.right.ix)
  -- | closed interval
  let len := ixr - ixl + 1
  let substr : Substring := (contents.toSubstring.drop ixl.toNat).take len.toNat
  let nspaces : Int := note.left.ix - ixl
  let underline : String :=   ("".pushn ' ' nspaces.toNat).push '^'
  vgroup [doc "---|" ++ note.kind ++ "|---" 
          , doc note.left ++ " " ++ substr.toString
          , doc note.left ++ " " ++ underline
          , doc "---"]
 
  

-- TOPLEVEL PARSER
-- ==============

-- https://github.com/leanprover/lean4/blob/master/tests/playground/file.lean
def main (xs: List String): IO Unit := do
  -- let path : System.FilePath :=  xs.head!
  let path :=  xs.head!
  let contents ← FS.readFile path;
  IO.println "FILE\n====\n"
  IO.println contents
  IO.println "\nEDSL TESTING\n============\n"
  IO.println opRgnAttr0
  IO.println "PARSING\n=======\n"
  let ns := []
  let (loc, ns, _, res) <-  (pop ()).runP locbegin ns contents
  IO.println (vgroup $ ns.map (note_add_file_content contents))
  match res with
   | Result.ok op => do
     IO.println "parse success:"
     IO.println op
   | Result.err err => do
      IO.println "***Parse Error:***"
      IO.println (note_add_file_content contents err)
   | Result.debugfail err =>  do
      IO.println "***Debug Error:***"
      IO.println (note_add_file_content contents err)
     
  return ()
