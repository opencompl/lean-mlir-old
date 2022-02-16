import MLIR.AST
import MLIR.Doc
import MLIR.P

open MLIR.AST
open MLIR.Doc
open MLIR.P


namespace MLIR.MLIRParser
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

partial def pattr : P AttrEntry := do
  eat_whitespace
  let name <- pident
  eat_whitespace
  pconsume '='
  let value <- pattrvalue
  return (AttrEntry.mk name value)

  

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
   let args <-  (if (<- ppeek? '(')
                then pintercalated '(' pblockoperand ',' ')'
                else return [])
   pconsume ':'
   let ops <- pmany1 (popcall u)
   -- pnote $ "pblock ops: " ++ List.toString ops
   return (BasicBlock.mk name args ops)

end   -- end the mutual block

theorem parse_of_print_id: 
  ∀ (o: Op), ((pop ()).runP locbegin [] (Pretty.doc o)).snd.snd.snd = Result.ok o := sorry

end MLIR.MLIRParser
