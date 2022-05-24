import MLIR.AST
import MLIR.Doc
import MLIR.P
import MLIR.Dialects.BuiltinModel

open MLIR.AST
open MLIR.Doc
open MLIR.P


namespace MLIR.MLIRParser

-- We can parse any dialect at least as large as builtin
section
variable {α σ ε} [δ: Dialect α σ ε] [CoeDialect builtin δ]

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
partial def pregion (u: Unit) : P (Region δ) :=  do
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


partial def pdim : P Dimension := do
  if (<- ppeek? '?')
  then return Dimension.Unknown
  else do
    let sz <- pnat
    return Dimension.Known sz

partial def ptype_vector : P (MLIRType builtin) := do
  pident? "vector"
  pconsume '<'
  let sz <- pnat
  pconsume 'x'
  let ty <- ptype ()
  pconsume '>'
  return builtin.vector [sz] [] ty

-- !<ident>.<ident>
partial def puser (u: Unit): P MLIRTy := do
    pconsume '!'
    let dialect <- pident
    pconsume '.'
    let ty <- pident
    return MLIRType.undefined (dialect ++ "." ++ ty)

partial def ptype (u: Unit) : P MLIRTy := do
  eat_whitespace
  let dom <- (match (<- ppeek) with
             | some '(' => do
                let args <- pintercalated '(' (ptype u) ',' ')'
                return MLIRType.tuple args
             | some 'i' => do
                 pconsume 'i'
                 let num <- pnumber
                 return MLIRType.int num
             | some '!' => do
                  puser ()
             | other => do
                perror ("uknown type starting with |" ++ toString other ++ "|."))
  eat_whitespace
  match (<- ppeek? '-')  with
  | true => do
        pconsume '-'
        pconsume '>' -- consume arrow
        let codom <- (ptype u)
        return MLIRType.fn dom codom
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
  return AttrValue.int num ty

partial def ptensorelem (u: Unit): P TensorElem := do
  if (<- ppeek? '[') then
    let ts <- pintercalated '[' (ptensorelem ()) ','  ']'
    return TensorElem.nested ts
  else
    let n <- pnumber
    return TensorElem.int n

partial def pattrvalue_dense (u: Unit): P (AttrValue δ) := do
  pident? "dense"
  pconsume '<'
  let v <- ptensorelem ()
  pconsume '>'
  pconsume ':'
  let ty <- ptype_vector
  return builtin.denseWithType v ty


partial def pattrvalue_list (u: Unit): P (AttrValue δ) := do
  let ts <- pintercalated '[' (pattrvalue ()) ','  ']'
  return AttrValue.list ts

partial def pattrvalue (u: Unit): P (AttrValue δ) := do
 pnote "hunting for attribute value"
 por pattrvalue_int $
 por (pmap AttrValue.str pstr) $
 por (pmap AttrValue.type (ptype ())) $
 por (pattrvalue_list ()) $
 (pattrvalue_dense ())

partial def pattr : P (AttrEntry δ) := do
  eat_whitespace
  let name <- pident
  eat_whitespace
  pconsume '='
  let value <- pattrvalue ()
  return (AttrEntry.mk name value)



partial def pop (u: Unit) : P (Op δ) := do
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


partial def popcall (u: Unit) : P (BasicBlockStmt δ) := do
   eat_whitespace
   if (<- ppeek? '%')
   then do
     let val <- pssaval
     pconsume '='
     let op <- pop u
     let index := none -- for syntax %val:ix = ...
     return (BasicBlockStmt.StmtAssign val index op)
   else do
     let op <- pop u
     return (BasicBlockStmt.StmtOp op)

-- | parse a sequence of ops, with no label
partial def pentryblock_no_label (u: Unit) : P (BasicBlock δ) := do
   let ops <- pmany1 (popcall u)
   pnote $ "pentry ops: " ++ List.toString ops
   return (BasicBlock.mk "entry" [] ops)



partial def pblock (u: Unit) : P (BasicBlock δ) := do
   pconsume '^'
   let name <- pident
   let args <-  (if (<- ppeek? '(')
                then pintercalated '(' pblockoperand ',' ')'
                else return [])
   pconsume ':'
   let ops <- pmany1 (popcall u)
   -- pnote $ "pblock ops: " ++ List.toString ops
   return (BasicBlock.mk name args ops)

end   -- end the mutual block

theorem parse_of_print_id:
  ∀ (o: Op δ), ((pop ()).runP locbegin [] (Pretty.doc o)).snd.snd.snd = Result.ok o := sorry

end   -- end of the section defining δ

end MLIR.MLIRParser
