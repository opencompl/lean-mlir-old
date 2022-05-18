import MLIR.AST
import MLIR.Doc
import MLIR.P
import MLIR.Dialects.BuiltinModel

open MLIR.AST
open MLIR.Doc
open MLIR.P


namespace MLIR.MLIRParser
-- | ^ <name>
def pbbname : P.{u} (ULift BBName) := do
  pconsume '^'
  let name <- pident
   return ULift.up (BBName.mk name.down)

-- | % <name>
partial def pssaval : P.{u} (ULift SSAVal) := do
  eat_whitespace
  pconsume '%'
  let name <- pident
  return ULift.up $ SSAVal.SSAVal name.down


mutual

-- | mh, needs to be mutual. Let's see if LEAN lets me do this.
partial def pregion (u: Unit) : P.{u+1} Region := do
  pconsume '{'
  -- HACK: entry block need not print block header. See: examples/region-with-no-args.mlir
  let b <- (if (<- ppeek? '^').down
           then pblock u 
           else pentryblock_no_label u)

  let bs <- (if (<- ppeek? '^').down
            then pmany1 (pblock u)
            else ppure [])
  pconsume '}'
  return (Region.mk (b::bs))


partial def pdim : P.{u+1} (ULift Dimension) := do
  if (<- ppeek? '?').down
  then return ULift.up Dimension.Unknown
  else do 
    let sz <- pnat
    return ULift.up $ Dimension.Known sz.down

partial def ptype_vector : P.{u+1} MLIRTy := do
  pident? "vector"
  pconsume '<'
  let sz <- pnumber
  pconsume 'x'
  let ty <- ptype ()
  pconsume '>'
  return MLIRTy.vector [sz.down] [] ty

-- !<ident>.<ident>  
partial def puser (u: Unit): P.{u+1} MLIRTy := do
    pconsume '!'
    let dialect <- pident
    pconsume '.'
    let ty <- pident
    let name := dialect.down ++ "." ++ ty.down
    return MLIRTy.undefined name

partial def ptype (u: Unit) : P.{u+1} MLIRTy := do
  eat_whitespace
  let dom <- (match (<- ppeek) with
             | some (ULift.up '(') => do
                let args <- pintercalated '(' (ptype u) ',' ')'
                return MLIRTy.tuple args
             | some (ULift.up 'i') => do
                 pconsume 'i'
                 let num <- pnumber
                 return MLIRTy.int num.down
             | some (ULift.up '!') => do
                  puser ()
             | other => do
                perror ("uknown type starting with |" ++ toString other ++ "|."))
  eat_whitespace
  match (<- ppeek? '-').down  with
  | true => do
        pconsume '-'
        pconsume '>' -- consume arrow
        let codom <- (ptype u)
        return MLIRTy.fn dom codom
  | false => do
     return dom

partial def pblockoperand : P.{u+1} (SSAVal × MLIRTy) := do
  eat_whitespace
  let operand <- pssaval
  pconsume ':'
  let ty <- (ptype ())
  return (operand.down, ty)


-- | either a string, or a type. Can be others.

partial def pattrvalue_int : P.{u+1} AttrVal := do
  let num <- pnumber
  pconsume ':'
  let ty <- ptype ()
  return AttrVal.int num.down ty

partial def ptensorelem (u: Unit): P.{u+1} (ULift TensorElem) := do
  if (<- ppeek? '[').down then
    let ts <- pintercalated '[' (ptensorelem ()) ','  ']'
    return ULift.up $ TensorElem.nested (ts.map (·.down))
  else 
    let n <- pnumber
    return ULift.up $ TensorElem.int n.down

partial def pattrvalue_dense (u: Unit): P.{u+1} AttrVal := do
  pident? "dense"
  pconsume '<'
  let v <- ptensorelem ()
  pconsume '>'
  pconsume ':'
  let ty <- ptype_vector
  return AttrVal.dense v.down ty


partial def pattrvalue_list (u: Unit): P.{u+1} AttrVal := do
  let ts <- pintercalated '[' (pattrvalue ()) ','  ']'
  return AttrVal.list ts

partial def pattrvalue (u: Unit): P.{u+1} AttrVal := do
 pnote "hunting for attribute value"
 por pattrvalue_int $ 
 por (pmap AttrVal.str (ULift.down <$> pstr)) $
 por (pmap AttrVal.type (ptype ())) $
 por (pattrvalue_list ()) $
 (pattrvalue_dense ())

partial def pattr : P.{u+1} AttrEntry := do
  eat_whitespace
  let name <- pident
  eat_whitespace
  pconsume '='
  let value <- pattrvalue ()
  return (AttrEntry.mk name.down value)

  

partial def pop (u: Unit) : P.{u+1} Op := do
  eat_whitespace
  match (<- ppeek) with 
  | some (ULift.up '\"') => do
    let name <- pstr
    let args <- pintercalated '(' pssaval ',' ')'
    let bbs <- (if (<- ppeek? '[' ).down
                then pintercalated '[' pbbname ','  ']'
                else return [])
    let hasRegion <- ppeek? '('
    let regions <- (if hasRegion.down
                      then pintercalated '(' (pregion ()) ',' ')'
                      else pure [])
    -- | parse attributes
    let hasAttrs <- ppeek? '{'
    let attrs <- (if hasAttrs.down
              then  pintercalated '{' pattr ',' '}' 
              else pure [])
    pconsume ':'
    let ty <- ptype u
    return (Op.mk  name.down (args.map (·.down)) (bbs.map (·.down)) regions attrs ty)
  | some (ULift.up '%') => perror "found %, don't know how to parse ops yet"
  | other => perror ("expected '\"' or '%' to begin operation definition. found: " ++ toString other)

partial def popcall (u: Unit) : P.{u+1} BasicBlockStmt := do
   let x := (← pssaval)
   eat_whitespace
   if (<- ppeek? '%').down
   then do
     let val <- pssaval
     pconsume '='
     let op <- pop u
     let index := none -- for syntax %val:ix = ...
     return (BasicBlockStmt.StmtAssign val.down index op)
   else do
     let op <- pop u
     return (BasicBlockStmt.StmtOp op)

-- | parse a sequence of ops, with no label
partial def pentryblock_no_label (u: Unit) : P.{u+1} BasicBlock := do
   let ops <- pmany1 (popcall u)
   pnote $ "pentry ops: " ++ List.toString ops
   return (BasicBlock.mk "entry" [] ops)


   
partial def pblock (u: Unit) : P.{u+1} BasicBlock := do
   pconsume '^'
   let name <- pident
   let args <-  (if (<- ppeek? '(').down
                then pintercalated '(' pblockoperand ',' ')'
                else return [])
   pconsume ':'
   let ops <- pmany1 (popcall u)
   -- pnote $ "pblock ops: " ++ List.toString ops
   return (BasicBlock.mk name.down args ops)

end   -- end the mutual block

theorem parse_of_print_id: 
  ∀ (o: Op), ((pop ()).runP locbegin [] (Pretty.doc o)).snd.snd.snd = Result.ok o := sorry

end MLIR.MLIRParser
