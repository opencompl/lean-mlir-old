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
inductive MLIRTy : Type where
| fn : MLIRTy -> MLIRTy -> MLIRTy
| int : Int -> MLIRTy
| tuple : List MLIRTy -> MLIRTy

inductive SSAVal : Type where
  | SSAVal : String -> SSAVal

inductive AttrVal : Type where
| str : String -> AttrVal
| type :MLIRTy -> AttrVal

inductive Attr : Type where
  | mk: (key: String) 
      -> (value: AttrVal)
      -> Attr

inductive Op : Type where 
 | mk: (name: String) 
      -> (args: List SSAVal)
      -> (attrs: List Attr)
      -> (region: List Region) 
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


partial def mlirty_to_string (ty: MLIRTy): String :=
  match ty with
  | MLIRTy.int k => "i" ++ (toString k)
  | MLIRTy.tuple ts => "(" ++ (intercalate ", " (ts.map mlirty_to_string)) ++ ")"
  | MLIRTy.fn dom codom => (mlirty_to_string dom) ++ " -> " ++ (mlirty_to_string codom)

instance : ToString MLIRTy := {
 toString := mlirty_to_string
}




def ssaval_to_doc (val: SSAVal): Doc := 
  match val with
  | SSAVal.SSAVal name => Doc.Text ("%" ++ name)

partial def intercalate_doc_rec_ (ds: List d) (f: d -> Doc) (i: Doc): Doc :=
  match ds with
  | [] => Doc.Text ""
  | (d::ds) => i ++ f d ++ intercalate_doc_rec_ ds f i

partial def intercalate_doc (ds: List d) (f: d -> Doc) (i: Doc): Doc :=
 match ds with
 | [] => Doc.Text ""
 | [d] => f d
 | (d::ds) => (f d) ++ intercalate_doc_rec_ ds f i


mutual
partial def op_to_doc (op: Op): Doc := 
    match op with
    | (Op.mk name args attrs rgns ty) => 
        let doc_name := (toString '"') ++ name ++ (toString '"')
        let doc_rgns := if List.isEmpty rgns then Doc.Text "" else " (" ++ Doc.Nest (Doc.VGroup (rgns.map rgn_to_doc)) ++ ")"
        let doc_ty := toString ty
        let doc_args := "(" ++ intercalate_doc args ssaval_to_doc ", " ++ ")"
        doc_name ++ doc_args ++  doc_rgns ++ " : " ++ doc_ty

partial def bb_stmt_to_doc (stmt: BasicBlockStmt): Doc :=
  match stmt with
  | BasicBlockStmt.StmtAssign lhs rhs => (ssaval_to_doc lhs) ++ " = " ++ (op_to_doc rhs)
  | BasicBlockStmt.StmtOp rhs => (op_to_doc rhs)

partial def bb_to_doc(bb: BasicBlock): Doc :=
  match bb with
  | (BasicBlock.mk name args stmts) => 
     let bbargs := if args.isEmpty then Doc.Text ""
                   else "(" ++ intercalate_doc args (fun (ssaval, ty) => ssaval_to_doc ssaval ++ ":" ++ mlirty_to_string ty) ", " ++ ")"
     let bbname := "^" ++ name ++ bbargs ++ ":"
     let bbbody := Doc.Nest (Doc.VGroup (stmts.map bb_stmt_to_doc))
     Doc.VGroup [bbname, bbbody]

partial def rgn_to_doc(rgn: Region): Doc :=
  match rgn with
  | (Region.mk bbs) => "{" ++ Doc.VGroup [Doc.Nest (Doc.VGroup (bbs.map bb_to_doc)), "}"]
 
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

 
def advance1 (l: Loc) (c: Char): Loc :=
  if c == '\n'
    then { line := l.line + 1, column := 1  }
    else return { line := l.line, column := l.column + 1}

-- | move a loc by a string.
partial def advance (l: Loc) (s: String): Loc :=
  if isEmpty s then l
  else let c := s.front; advance (advance1 l c) (s.drop 1)

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


-- | TODO: enable notes, refactor type into Loc x String x [Note] x (Result ParseError a)
structure P (a: Type) where 
   runP: Loc -> String -> Result ParseError (Loc × String × a)



-- | map for parsers
def pmap (f : a -> b) (pa: P a): P b := {
  runP :=  λ loc s => 
    match pa.runP loc s with
      | Result.ok (l', s', a) => Result.ok (l', s', f a)
      | Result.err e => Result.err e
}


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

-- try p. if success, return value. if not, run q
def por (p: P a) (q: P a) : P a :=  {
  runP := λ loc s => 
    match p.runP loc s with
      | Result.ok a => Result.ok a 
      | Result.err _ => q.runP loc s
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
  | some c' => 
     if c == c' then padvance_char_INTERNAL c
     else perror ("pconsume: expected character |" ++ toString c ++ "|. Found |" ++ toString c' ++ "|.")
  | none =>  perror ("pconsume: expected character |" ++ toString c ++ "|. Found EOF")


def ppeek?(c: Char) : P Bool := do
  let cm <- ppeek
  return (cm == some c)


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
  ptakewhile (fun c => (c != ' ' && c != '\t' && c != '\n') && (isAlphanum c || c == '_'))

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
               else perror ("intercalate: expected |" ++ i.toString ++ "|  or |" ++ r.toString ++ "|, found |" ++ c.toString ++ "|.")
   | _ =>  perror ("intecalate: expected |" ++ i.toString ++ "|  or |" ++ r.toString ++ "|, found EOF" )


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


-- | parse <p>+ for a given <p>
partial def pmany1 (p: P a) : P (List a) := do
  let a1 <- p
  let as <- por (pmany1 p) (psuccess [])
  return (a1::as)

mutual


-- | mh, needs to be mutual. Let's see if LEAN lets me do this.
partial def pregion (u: Unit) : P Region :=  do
  pconsume '{'
  -- HACK: entry block need not print block header. See: examples/region-with-no-args.mlir
  let b <- if (<- ppeek? '^')
           then pblock u 
           else pentryblock_no_label u -- TODO: make this many
  pconsume '}'
  return (Region.mk [b])


partial def pssaval : P SSAVal := do
  eat_whitespace
  pconsume '%'
  let name <- pident
  return (SSAVal.SSAVal name)


partial def ptype (u: Unit) : P MLIRTy := do
  eat_whitespace
  let dom <- (match (<- ppeek) with
             | some '(' => do
                let args <- pintercalated '(' (ptype u) ',' ')'
                return MLIRTy.tuple args
             | some 'i' => do
                 pconsume 'i'
                 let _ <- pident -- HACK: we should actually consume a number
                 return MLIRTy.int 42
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
partial def pattrvalue : P AttrVal := do
 por (pmap AttrVal.str pstr) (pmap AttrVal.type (ptype ()))

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
    return (Op.mk  name args [] regions ty)
  | some '%' => perror "found %, don't know how to parse ops yet"
  | other => perror ("expected '\"' or '%' to begin operation definition. found: " ++ toString other)


partial def popcall (u: Unit) : P BasicBlockStmt := do
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
   return (BasicBlock.mk "entry" [] ops)


   
partial def pblock (u: Unit) : P BasicBlock := do
   pconsume '^'
   let name <- pident
   let args <- pintercalated '(' pblockoperand ',' ')'
   pconsume ':'
   let ops <- pmany1 (popcall u)
   return (BasicBlock.mk name args ops)
end  

-- EDSL
-- ====

declare_syntax_cat mlir_bb_line
declare_syntax_cat mlir_op_results
declare_syntax_cat mlir_op_call
declare_syntax_cat mlir_op_call_args
declare_syntax_cat mlir_op_call_type
declare_syntax_cat mlir_op_operand
declare_syntax_cat mlir_type


syntax mlir_op_call : mlir_bb_line
syntax mlir_op_results "=" mlir_op_call  : mlir_bb_line
syntax strLit mlir_op_call_args ":" mlir_op_call_type : mlir_op_call -- no region


syntax "(" ")" : mlir_op_call_args
syntax "(" mlir_op_operand ")" : mlir_op_call_args
syntax "(" mlir_op_operand "," mlir_op_operand","* ")" : mlir_op_call_args


-- EDSL OPERANDS
-- ==============

syntax "%" ident : mlir_op_operand

syntax "mlir_op_operand% " mlir_op_operand : term -- translate mlir_op_call into term
macro_rules
  | `(mlir_op_operand% % $x:ident) => `(SSAVal.SSAVal $(Lean.quote (toString x.getId))) 

def xx := (mlir_op_operand% %x)
def xxx := (mlir_op_operand% %x)
#print xx
#print xxx


-- EDSL OP CALL
-- ============

syntax "mlir_op_call_args% " mlir_op_call_args : term -- translate mlir_op_call into term
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
  let res := (pop ()).runP locbegin contents
  match res with
   | Result.ok (loc, str, op) => IO.println op
   | Result.err res => IO.println res
  return ()
