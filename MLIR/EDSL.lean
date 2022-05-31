import Lean
import Lean.Elab
import Lean.Meta
import Lean.Parser
import Lean.PrettyPrinter
import Lean.PrettyPrinter.Formatter
import MLIR.AST
import Lean.Parser
import Lean.Parser.Extra
-- import Lean.Init.Meta


open Lean
open Lean.Parser
open Lean.Elab
open Lean.Meta
open Lean.Parser
open Lean.Parser.ParserState
open Lean.PrettyPrinter
open Lean.PrettyPrinter.Formatter

open MLIR.AST

namespace MLIR.EDSL


-- | Custom parsers for balanced brackets
inductive Bracket
| Square -- []
| Round -- ()
| Curly -- {}
| Angle -- <>
deriving Inhabited, DecidableEq

instance : ToString Bracket where
   toString :=
    fun b =>
     match b with
     | .Square => "["
     | .Round => "("
     | .Curly => "{"
     | .Angle => "<"


-- TODO: remove <Tab> from quail
def isOpenBracket(c: Char): Option Bracket :=
match c with
| '(' => some .Round
| '[' => some .Square
| '{' => some .Curly
| '<' => some .Angle
| _ => none

def isCloseBracket(c: Char):Option Bracket :=
match c with
| ')' => some .Round
| ']' => some .Square
| '{' => some .Curly
| '<' => some .Angle
| _ => none

mutual

#check ParserState

#check Format

-- 'a -> symbol
-- `a -> antiquotation `(... ,(...))
partial def consumeCloseBracket(c: Bracket)
  (startPos: String.Pos)
  (i: String.Pos)
  (input: String)
  (brackets: List Bracket)
  (ctx: ParserContext)
  (s: ParserState): ParserState := Id.run do
    -- dbg_trace "consumeCloseBracket"
    match brackets with
    | b::bs =>
      if b == c
      then
        if bs == []
        then
          -- dbg_trace f!"closed brackets at {i}"
          let parser_fn := Lean.Parser.mkNodeToken `balanced_brackets startPos
          parser_fn ctx (s.setPos (input.next i)) -- consume the input here.
        else balancedBracketsFnAux startPos (input.next i) input bs ctx s
      else s.mkError $ "| found Opened `" ++ toString b ++ "` expected to close at `" ++ toString c ++ "`"
    | _ => s.mkError $ "| found Closed `" ++ toString c ++ "`, but have no opened brackets on stack"


partial def balancedBracketsFnAux (startPos: String.Pos)
  (i: String.Pos)
  (input: String)
  (bs: List Bracket) (ctx: ParserContext) (s: ParserState): ParserState :=
  if input.atEnd i
  then s.mkError "fonud EOF"
  else
  match input.get i with
  -- opening parens
  | '(' => balancedBracketsFnAux startPos (input.next i) input (Bracket.Round::bs) ctx s
  | '[' => balancedBracketsFnAux startPos (input.next i) input (Bracket.Square::bs) ctx s
  | '<' => balancedBracketsFnAux startPos (input.next i) input (Bracket.Angle::bs) ctx s
  | '{' => balancedBracketsFnAux startPos (input.next i) input (Bracket.Curly::bs) ctx s
  -- closing parens
  | ')' => consumeCloseBracket Bracket.Round startPos i input bs ctx s
  | ']' => consumeCloseBracket Bracket.Square startPos i input bs ctx s
  | '>' => consumeCloseBracket Bracket.Angle startPos i input bs ctx s
  | '}' => consumeCloseBracket Bracket.Curly startPos i input bs ctx s
  | c => balancedBracketsFnAux startPos (input.next i) input bs ctx s

end

-- | TODO: filter tab complete by type?
def balancedBracketsFnEntry (ctx: ParserContext) (s: ParserState): ParserState :=
  if ctx.input.get s.pos == '<'
  then balancedBracketsFnAux
   (startPos := s.pos)
   (i := s.pos)
   (input := ctx.input)
   (bs := [])
   ctx s
  else s.mkError "Expected '<'"


@[inline]
def balancedBrackets : Parser :=
   withAntiquot (mkAntiquot "balancedBrackets" `balancedBrackets) {
       fn := balancedBracketsFnEntry,
       info := mkAtomicInfo "balancedBrackets" : Parser
    }

#check balancedBrackets


-- Code stolen from test/WebServer/lean
@[combinatorFormatter MLIR.EDSL.balancedBrackets]
def MLIR.EDSL.balancedBrackets.formatter : Formatter := pure ()

@[combinatorParenthesizer MLIR.EDSL.balancedBrackets]
def MLIR.EDSL.balancedBracketsParenthesizer : Parenthesizer := pure ()


macro "[balanced_brackets|" xs:balancedBrackets "]" : term => do
  match xs[0] with
  | .atom _ val => return Lean.quote val
  | _  => Macro.throwError "expected balanced bracts to have atom"


def testBalancedBrackets : String := [balanced_brackets| < { xxasdasd } > ]
#print testBalancedBrackets



-- | positive and negative numbers, hex, octal
declare_syntax_cat mlir_int
syntax numLit: mlir_int

def IntToString (i: Int): String := i.repr

instance : Quote Int := ⟨fun n => Syntax.mkNumLit <| n.repr⟩

def quoteMDimension (d: Dimension): MacroM Syntax :=
  match d with
  | Dimension.Known n => do 
    `(Dimension.Known $(quote n))
  | Dimension.Unknown => `(Dimension.Unknown)


def quoteMList (k: List Syntax) (ty: Syntax): MacroM Syntax :=
  match k with 
  | [] => `(@List.nil $ty)
  | (k::ks) => do
      let sks <- quoteMList ks ty
      `([$k] ++ $sks)


-- AFFINE SYTAX
-- ============
 
declare_syntax_cat affine_expr
declare_syntax_cat affine_tuple
declare_syntax_cat affine_map 


syntax ident : affine_expr
syntax "(" sepBy(affine_expr, ",") ")" : affine_tuple
syntax "affine_map<" affine_tuple "->" affine_tuple ">" : affine_map

syntax "[affine_expr|" affine_expr "]" : term
syntax "[affine_tuple|" affine_tuple "]" : term 
syntax "[affine_map|" affine_map "]" : term  
-- syntax "[affine_map|" affine_map "]" : term  

macro_rules 
| `([affine_expr| $xraw:ident ]) => do 
  let xstr := xraw.getId.toString
  `(AffineExpr.Var $(Lean.quote xstr))

macro_rules
| `([affine_tuple| ( $xs,* ) ]) => do
   let initList  <- `(@List.nil MLIR.AST.AffineExpr)
   let argsList <- xs.getElems.foldlM
    (init := initList) 
    (fun xs x => `($xs ++ [[affine_expr| $x]]))
   `(AffineTuple.mk $argsList)
   
  
macro_rules
| `([affine_map| affine_map< $xs:affine_tuple -> $ys:affine_tuple >]) => do
  let xs' <- `([affine_tuple| $xs])
  let ys' <- `([affine_tuple| $ys])
  `(AffineMap.mk $xs' $ys' )
 

-- EDSL
-- ====

declare_syntax_cat mlir_bb
declare_syntax_cat mlir_entry_bb
declare_syntax_cat mlir_region
declare_syntax_cat mlir_bb_stmt
declare_syntax_cat mlir_bb_stmts
declare_syntax_cat mlir_op_results
declare_syntax_cat mlir_op
declare_syntax_cat mlir_op_args
declare_syntax_cat mlir_op_successor_args
declare_syntax_cat mlir_op_type
declare_syntax_cat mlir_op_operand
declare_syntax_cat mlir_type

-- syntax strLit mlir_op_args ":" mlir_op_type : mlir_op -- no region
-- 


-- EDSL OPERANDS
-- ==============

syntax "%" numLit : mlir_op_operand

syntax "%" ident : mlir_op_operand

syntax "[mlir_op_operand|" mlir_op_operand "]" : term
macro_rules
  | `([mlir_op_operand| $$($q)]) => return q
  | `([mlir_op_operand| % $x:ident]) => `(SSAVal.SSAVal $(Lean.quote (x.getId.toString))) 
  | `([mlir_op_operand| % $n:num]) => `(SSAVal.SSAVal (IntToString $n))

def operand0 := [mlir_op_operand| %x]
#print operand0

def operand1 := [mlir_op_operand| %x]
#print operand1

def operand2 := [mlir_op_operand| %0]
#print operand2


-- EDSL OP-SUCCESSOR-ARGS
-- =================

-- successor-list       ::= `[` successor (`,` successor)* `]`
-- successor            ::= caret-id (`:` bb-arg-list)?

declare_syntax_cat mlir_op_successor_arg -- bb argument
syntax "^" ident : mlir_op_successor_arg -- bb argument with no operands
-- syntax "^" ident ":" "(" mlir_op_operand","* ")" : mlir_op_successor_arg

syntax "[mlir_op_successor_arg|" mlir_op_successor_arg "]" : term

macro_rules
  | `([mlir_op_successor_arg| ^ $x:ident  ]) => 
      `(BBName.mk $(Lean.quote (x.getId.toString)))

def succ0 :  BBName := ([mlir_op_successor_arg| ^bb])
#print succ0


-- EDSL MLIR TYPES
-- ===============


syntax "[mlir_type|" mlir_type "]" : term

syntax "(" mlir_type,* ")" : mlir_type
macro_rules
| `([mlir_type| ( $xs,* )]) => do
      let xs <- xs.getElems.mapM (fun x => `([mlir_type| $x]))
      let x <- quoteMList xs.toList (<- `(MLIRTy))
      `(MLIRTy.tuple $x)

-- syntax "(" mlir_type ")" : mlir_type
-- syntax "(" mlir_type "," mlir_type ")" : mlir_type
-- | HACK: just switch to real parsing of lists
-- syntax "(" mlir_type "," mlir_type "," mlir_type ")" : mlir_type
syntax mlir_type "->" mlir_type : mlir_type
syntax "{{" term "}}" : mlir_type
syntax "!" str : mlir_type
syntax "!" ident : mlir_type
syntax ident: mlir_type


set_option hygiene false in -- allow i to expand 
macro_rules
  | `([mlir_type| $x:ident ]) => do
        let xstr := x.getId.toString
        if xstr == "index"
        then
          `(MLIRTy.index)
        else if xstr.front == 'i' || xstr.front == 'f'
        then do 
          let xstr' := xstr.drop 1
          match xstr'.toInt? with
          | some i => 
            let lit := Lean.Syntax.mkNumLit xstr'
            if xstr.front == 'i'
            then `(MLIRTy.int $lit)
            else `(MLIRTy.float $lit)
          | none => 
              Macro.throwError $ "cannot convert suffix of i/f to int: " ++ xstr
        else Macro.throwError $ "expected i<int> or f<int>, found: " ++ xstr  -- `(MLIRTy.int 1337)

macro_rules
| `([mlir_type| ! $x:str ]) => `(MLIRTy.user $x)

macro_rules
| `([mlir_type| ! $x:ident ]) => `(MLIRTy.user $(Lean.quote x.getId.toString))

def tyIndex : MLIRTy := [mlir_type| index]
#eval tyIndex

def tyUser : MLIRTy := [mlir_type| !"lz.int"]
#eval tyUser

def tyUserIdent : MLIRTy := [mlir_type| !shape.value]
#eval tyUserIdent


def tyi32NoGap : MLIRTy := [mlir_type| i32]
#eval tyi32NoGap
def tyf32NoGap : MLIRTy := [mlir_type| f32]
#eval tyf32NoGap

macro_rules
| `([mlir_type| {{ $t }} ]) => return t -- antiquot type

-- macro_rules
--   | `([mlir_type| ( ) ]) => `(MLIRTy.tuple [])
--   | `([mlir_type| ( $x:mlir_type )]) => 
--         `(MLIRTy.tuple [ [mlir_type|$x] ])
--   | `([mlir_type| ( $x:mlir_type, $y:mlir_type )]) => 
--     `(MLIRTy.tuple [ [mlir_type|$x], [mlir_type|$y] ] )
--   | `([mlir_type| ( $x:mlir_type, $y:mlir_type, $z:mlir_type )]) => 
--     `(MLIRTy.tuple [ [mlir_type|$x], [mlir_type|$y], [mlir_type| $z ] ] )

macro_rules
  | `([mlir_type| $dom:mlir_type -> $codom:mlir_type]) =>
     `(MLIRTy.fn [mlir_type|$dom] [mlir_type|$codom])

def ty0 : MLIRTy := [mlir_type| ( )]
def tyi32 : MLIRTy := [mlir_type| i32] -- TODO: how to keep no gap?
-- def tyi32' : MLIRTy := ([mlir_type| i32) -- TODO: how to keep no gap?
def tysingle : MLIRTy := [mlir_type| (i42)]
def typair : MLIRTy := [mlir_type| (i32, i64)]
def tyfn0 : MLIRTy := [mlir_type| () -> ()]
def tyfn1 : MLIRTy := [mlir_type| (i11) -> (i12)]
def tyfn2 : MLIRTy := [mlir_type| (i21, i22) -> (i23, i24)]
def tyfn3 : MLIRTy := [mlir_type| (i21, i22, i23) -> (i23, i24, i25)]
#print ty0
#print tyi32
#print typair
#print tyfn0
#print tyfn1
-- #print tyi32'


declare_syntax_cat mlir_dimension

syntax "?" : mlir_dimension
syntax num : mlir_dimension

syntax "[mlir_dimension|" mlir_dimension "]" : term
macro_rules
| `([mlir_dimension| ?]) => `(Dimension.Unknown)
macro_rules
| `([mlir_dimension| $x:num ]) => 
    `(Dimension.Known $x)

def dim0 := [mlir_dimension| 30]
#print dim0

def dim1 := [mlir_dimension| ?]
#print dim1


-- | 1 x 2 x 3 x ..
declare_syntax_cat mlir_dimension_list
syntax (mlir_dimension "×")* mlir_type : mlir_dimension_list

def string_to_dimension (s: String): MacroM Dimension := do
  if s == "?"
  then return Dimension.Unknown
  else if s.isNat
  then return Dimension.Known s.toNat!
  else Macro.throwError ("unknown dimension: | " ++ s ++ "  |")


-- (MLIR.EDSL.«mlir_dimension_list_×_»
--  [
--    [(MLIR.EDSL.mlir_dimension_ (numLit "3")) "×"]
--    [(MLIR.EDSL.mlir_dimension_ (numLit "3")) "×"]]
 -- (MLIR.EDSL.mlir_type__ `i32))| )

-- | TODO: assert that the string we get is of the form x3x4x?x2...
-- that is, interleaved x and other stuff.
def parseTensorDimensionList (k: Syntax) : MacroM (Syntax × Syntax) := do

  let ty <- `([mlir_type|  $(k.getArgs.back)])
  let dimensions := (k.getArg 0)
  let dimensions <- dimensions.getArgs.toList.mapM (fun x => `([mlir_dimension| $(x.getArg 0)]))
  let dimensions <- quoteMList dimensions (<- `(MLIR.AST.Dimension))
  -- Macro.throwError $ ("unknown dimension list:\n|" ++ (toString k.getArgs) ++ "|" ++ "\nDIMS: " ++ (toString dimensions) ++ " |\nTYPE: " ++ (toString ty)++ "")
  return (dimensions, ty)


  --       let xstr := dims.getId.toString
  --       let xparts := (xstr.splitOn "x").tail!
  --       let ty := xparts.getLast!
  --       let xparts := xparts.dropLast
  --       let xparts := [] ++ xparts -- TODO: add k into this list.
  --       -- Macro.throwError $ ("unknown dimension list: |" ++ (toString xparts) ++ "| )")

  --       let tyIdent := Lean.mkIdent ty
  --       -- let tyStx <- `([mlir_type|  $(quote tyIdent)])
  --       let tyStx <-  `([mlir_type|  i32])
  --       let dims <- xparts.mapM string_to_dimension
  --       let dimsStx <- quoteMList ([k] ++ (<- dims.mapM quoteMDimension))
  --       return (dimsStx, tyStx)
  -- -- | err => Macro.throwError $  ("unknown dimension list: |" ++ err.reprint.getD "???" ++ "| )")

-- === VECTOR TYPE ===
-- TODO: where is vector type syntax defined?
-- | TODO: fix bug that does not allow a trailing times.

-- static-dim-list ::= decimal-literal (`x` decimal-literal)*
-- | Encoding lookahead with notFollowedBy
declare_syntax_cat static_dim_list
syntax sepBy(numLit, "×", "×" notFollowedBy(mlir_type <|> "[")) : static_dim_list


syntax "[static_dim_list|" static_dim_list "]" : term
macro_rules
| `([static_dim_list| $[ $ns:num ]×* ]) => do
      quoteMList ns.toList (<- `(Int))

-- vector-dim-list := (static-dim-list `x`)? (`[` static-dim-list `]` `x`)?
declare_syntax_cat vector_dim_list
syntax (static_dim_list "×" ("[" static_dim_list "]" "×")? )? : vector_dim_list
-- vector-element-type ::= float-type | integer-type | index-type
-- vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
syntax "vector" "<" vector_dim_list mlir_type ">"  : mlir_type

set_option hygiene false in -- allow i to expand 
macro_rules
| `([mlir_type| vector < $[$fixed?:static_dim_list × $[ [ $scaled?:static_dim_list ] × ]? ]? $t:mlir_type  >]) => do
      let fixedDims <- match fixed? with  
        | some s =>  `([static_dim_list| $s])
        | none => `((@List.nil Int))
      let scaledDims <- match scaled? with  
        | some (some s) => `([static_dim_list| $s])
        | _ => `((@List.nil Int))
      `(MLIRTy.vector $fixedDims $scaledDims [mlir_type| $t])

def staticDimList0 : List Int := [static_dim_list| 1]
#reduce staticDimList0

def staticDimList1 : List Int := [static_dim_list| 1 × 2]
#reduce staticDimList1



def vectorTy0 := [mlir_type| vector<i32>]
#print vectorTy0

def vectorTy1 := [mlir_type| vector<2 × i32>]
#print vectorTy1

def vectorTy2 := [mlir_type| vector<2 × 3 × [ 4 ] × i32>]
#print vectorTy2


-- | TODO: is this actually necessary?
-- syntax  "<" mlir_dimension_list  ">"  : mlir_type
-- macro_rules
-- | `([mlir_type|  < $dims:mlir_dimension_list  >]) => do
--     let (dims, ty) <- parseTensorDimensionList dims 
--     `(MLIRTy.vector $dims $ty)


-- | TODO: fix bug that does not allow a trailing times.

syntax "tensor" "<"  mlir_dimension_list  ">"  : mlir_type
macro_rules
| `([mlir_type| tensor < $dims:mlir_dimension_list  >]) => do
    let (dims, ty) <- parseTensorDimensionList dims 
    `(MLIRTy.tensorRanked $dims $ty)

-- | TODO: this is a huge hack.
-- | TODO: I should be able to use the lower level parser to parse this cleanly?
syntax "tensor" "<"  "*" "×" mlir_type ">"  : mlir_type
syntax "tensor" "<*" "×" mlir_type ">"  : mlir_type
syntax "tensor" "<*×" mlir_type ">"  : mlir_type

macro_rules
| `([mlir_type| tensor < *× $ty:mlir_type >]) => do
    `(MLIRTy.tensorUnranked [mlir_type| $ty])

macro_rules
| `([mlir_type| tensor < * × $ty:mlir_type >]) => do
    `(MLIRTy.tensorUnranked [mlir_type| $ty])

macro_rules
| `([mlir_type| tensor <* × $ty:mlir_type >]) => do
    `(MLIRTy.tensorUnranked [mlir_type| $ty])

macro_rules
| `([mlir_type| tensor <*×$ty:mlir_type >]) => do
    `(MLIRTy.tensorUnranked [mlir_type| $ty])

def tensorTy0 := [mlir_type| tensor<3×3×i32>]
#print tensorTy0

def tensorTy1 := [mlir_type| tensor< * × i32>]
#print tensorTy1

def tensorTy2 := [mlir_type| tensor< * × f32>]
#print tensorTy2

def tensorTy3 := [mlir_type| tensor<*× f32>]
#print tensorTy3

def tensorTy4 := [mlir_type| tensor<* × f32>]
#print tensorTy4



-- EDSL MLIR USER ATTRIBUTES
-- =========================



-- EDSL MLIR OP CALL, MLIR BB STMT
-- ===============================

-- syntax strLit mlir_op_args ":" mlir_type : mlir_op

syntax "[mlir_op|" mlir_op "]" : term



syntax mlir_op: mlir_bb_stmt
syntax mlir_op_operand "=" mlir_op : mlir_bb_stmt
syntax mlir_op_operand ":" numLit "=" mlir_op : mlir_bb_stmt

syntax "[mlir_bb_stmt|" mlir_bb_stmt "]" : term

macro_rules
  | `([mlir_bb_stmt| $$($q)]) => `(coe $q)

macro_rules
  | `([mlir_bb_stmt| $call:mlir_op ]) =>
       `(BasicBlockStmt.StmtOp ([mlir_op| $call]))
  | `([mlir_bb_stmt| $res:mlir_op_operand = $call:mlir_op]) => 
       `(BasicBlockStmt.StmtAssign ([mlir_op_operand| $res]) none ([mlir_op| $call]))
  | `([mlir_bb_stmt| $res:mlir_op_operand : $ix:num = $call:mlir_op]) => 
       `(BasicBlockStmt.StmtAssign ([mlir_op_operand| $res]) (some $ix) ([mlir_op| $call]))



-- EDSL MLIR BASIC BLOCK OPERANDS
-- ==============================

declare_syntax_cat mlir_bb_operand
syntax mlir_op_operand ":" mlir_type : mlir_bb_operand

syntax "[mlir_bb_operand|" mlir_bb_operand "]" : term

macro_rules 
| `([mlir_bb_operand| $name:mlir_op_operand : $ty:mlir_type ]) => 
     `( ([mlir_op_operand| $name], [mlir_type|$ty]) ) 



-- EDSL MLIR BASIC BLOCKS
-- ======================



syntax "^" ident ":" mlir_bb_stmts : mlir_bb
syntax "^" ident "(" sepBy(mlir_bb_operand, ",") ")" ":" mlir_bb_stmts : mlir_bb

syntax (mlir_bb_stmt)* : mlir_bb_stmts

syntax "[mlir_bb_stmts|" mlir_bb_stmts "]" : term
macro_rules
| `([mlir_bb_stmts| $[ $stmts ]*  ]) => do
      let initList <- `(@List.nil MLIR.AST.BasicBlockStmt)
      stmts.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_bb_stmt|$x]])

macro_rules
  | `([mlir_bb_stmts| $$($q)]) => `(coe $q)



syntax "[mlir_bb|" mlir_bb "]": term


macro_rules 
| `([mlir_bb| ^ $name:ident ( $operands,* ) : $stmts ]) => do
   let initList <- `(@List.nil (MLIR.AST.SSAVal × MLIR.AST.MLIRTy))
   let argsList <- operands.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_bb_operand| $x]])
   let opsList <- `([mlir_bb_stmts| $stmts])
   `(BasicBlock.mk $(Lean.quote (name.getId.toString)) $argsList $opsList)
| `([mlir_bb| ^ $name:ident : $stmts ]) => do
   let opsList <- `([mlir_bb_stmts| $stmts])
   `(BasicBlock.mk $(Lean.quote (name.getId.toString)) [] $opsList)


-- ENTRY BB
-- ========


syntax mlir_bb : mlir_entry_bb
syntax mlir_bb_stmts : mlir_entry_bb
syntax "[mlir_entry_bb|" mlir_entry_bb "]" : term


macro_rules 
| `([mlir_entry_bb| $stmts:mlir_bb_stmts ]) => do
   let opsList <- `([mlir_bb_stmts| $stmts])
   `(BasicBlock.mk "entry" [] $opsList)

macro_rules 
| `([mlir_entry_bb| $bb:mlir_bb ]) => `([mlir_bb| $bb])

-- EDSL MLIR REGIONS
-- =================

syntax "{" (ws mlir_entry_bb ws)? (ws mlir_bb ws)* "}": mlir_region
syntax "[mlir_region|" mlir_region "]" : term

-- | map a macro on a list

macro_rules
| `([mlir_region| { $[ $entrybb ]? $[ $bbs ]* } ]) => do
   let initList <- match entrybb with 
                  | some entry => `([[mlir_entry_bb| $entry]])
                  | none => `([])
   let bbsList <- bbs.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_bb|$x]])
   `(Region.mk $bbsList)

macro_rules
| `([mlir_region| $$($q) ]) => return q


-- TENSOR LITERAL
-- ==============

declare_syntax_cat mlir_tensor
syntax numLit : mlir_tensor
syntax scientificLit : mlir_tensor

syntax "[" sepBy(mlir_tensor, ",") "]" : mlir_tensor

syntax ident: mlir_tensor
syntax "[mlir_tensor|" mlir_tensor "]" : term

macro_rules
| `([mlir_tensor| $x:num ]) => `(TensorElem.int $x)

macro_rules
| `([mlir_tensor| $x:scientific ]) => `(TensorElem.float $x)

macro_rules 
| `([mlir_tensor| $x:ident ]) => do 
      let xstr := x.getId.toString
      if xstr == "true" 
      then `(TensorElem.bool true)
      else if xstr == "false"
      then `(TensorElem.bool false)
      else Macro.throwError ("unknown tensor value: |" ++ xstr ++ "|")

macro_rules
| `([mlir_tensor| [ $xs,* ] ]) => do 
    let initList <- `([])
    let vals <- xs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_tensor| $x]]) 
    `(TensorElem.nested $vals)


def tensorValNum := [mlir_tensor| 42]
def tensorValFloat := [mlir_tensor| 0.000000]
def tensorValTrue := [mlir_tensor| true]
def tensorValFalse := [mlir_tensor| false]

-- MLIR ATTRIBUTE VALUE
-- ====================

-- | TODO: consider renaming this to mlir_attr
declare_syntax_cat mlir_attr_val
declare_syntax_cat mlir_attr_val_symbol
syntax "@" ident : mlir_attr_val_symbol
syntax "@" str : mlir_attr_val_symbol
syntax "#" ident : mlir_attr_val -- alias
syntax "#" strLit : mlir_attr_val -- aliass

syntax "#" ident "<" strLit ">" : mlir_attr_val -- opaqueAttr
syntax "#opaque<" ident "," strLit ">" ":" mlir_type : mlir_attr_val -- opaqueElementsAttr
syntax mlir_attr_val_symbol "::" mlir_attr_val_symbol : mlir_attr_val_symbol


declare_syntax_cat balanced_parens  -- syntax "#" ident "." ident "<" balanced_parens ">" : mlir_attr_val -- generic user attributes


syntax str: mlir_attr_val
syntax mlir_type : mlir_attr_val
syntax affine_map : mlir_attr_val
syntax mlir_attr_val_symbol : mlir_attr_val
syntax num (":" mlir_type)? : mlir_attr_val
syntax scientificLit (":" mlir_type)? : mlir_attr_val
syntax ident: mlir_attr_val

syntax "[" sepBy(mlir_attr_val, ",") "]" : mlir_attr_val
syntax "[mlir_attr_val|" mlir_attr_val "]" : term
syntax "[mlir_attr_val_symbol|" mlir_attr_val_symbol "]" : term

macro_rules
| `([mlir_attr_val| $$($x) ]) => `($x)

macro_rules
| `([mlir_attr_val|  $x:num ]) => `(AttrVal.int $x (MLIRTy.int 64))
| `([mlir_attr_val| $x:num : $t:mlir_type]) => `(AttrVal.int $x [mlir_type| $t])

macro_rules
| `([mlir_attr_val| true ]) => `(AttrVal.bool True)
| `([mlir_attr_val| false ]) => `(AttrVal.bool False)


macro_rules
| `([mlir_attr_val| # $dialect:ident < $opaqueData:str > ]) => do
  let dialect := Lean.quote dialect.getId.toString
  `(AttrVal.opaque $dialect $opaqueData)

macro_rules
| `([mlir_attr_val| #opaque< $dialect:ident, $opaqueData:str> : $t:mlir_type ]) => do
  let dialect := Lean.quote dialect.getId.toString
  `(AttrVal.opaqueElementsAttr $dialect $opaqueData $t)

macro_rules 
  | `([mlir_attr_val| $s:str]) => `(AttrVal.str $s)
  | `([mlir_attr_val| [ $xs,* ] ]) => do 
        let initList <- `([])
        let vals <- xs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_attr_val| $x]]) 
        `(AttrVal.list $vals)
  | `([mlir_attr_val| $i:ident]) => `(AttrVal.type [mlir_type| $i:ident])
  | `([mlir_attr_val| $ty:mlir_type]) => `(AttrVal.type [mlir_type| $ty])


syntax "dense<" mlir_tensor  ">" ":" mlir_type : mlir_attr_val
macro_rules
| `([mlir_attr_val| dense< $v:mlir_tensor > : $t:mlir_type]) => 
    `(AttrVal.dense [mlir_tensor| $v] [mlir_type| $t])

syntax "dense<" ">" ":" mlir_type: mlir_attr_val
macro_rules
| `([mlir_attr_val| dense< > : $t:mlir_type]) => 
    `(AttrVal.dense TensorElem.empty [mlir_type| $t])

macro_rules
  | `([mlir_attr_val| $a:affine_map]) =>
      `(AttrVal.affine [affine_map| $a])

macro_rules
| `([mlir_attr_val_symbol| @ $x:str ]) =>
      `(AttrVal.symbol $x)

macro_rules
| `([mlir_attr_val_symbol| @ $x:ident ]) =>
      `(AttrVal.symbol $(Lean.quote x.getId.toString))

macro_rules
| `([mlir_attr_val_symbol| $x:mlir_attr_val_symbol :: $y:mlir_attr_val_symbol ]) =>
      `(AttrVal.nestedsymbol [mlir_attr_val_symbol| $x] [mlir_attr_val_symbol| $y])


macro_rules
| `([mlir_attr_val| $x:mlir_attr_val_symbol ]) => `([mlir_attr_val_symbol| $x])



def attrVal0Str : AttrVal := [mlir_attr_val| "foo"]
#reduce attrVal0Str

def attrVal1Ty : AttrVal := [mlir_attr_val| (i32, i64) -> i32]
#reduce attrVal1Ty

def attrVal1bTy : AttrVal := [mlir_attr_val| i32]
#reduce attrVal1bTy

def attrVal2List : AttrVal := [mlir_attr_val| ["foo", "foo"] ]
#reduce attrVal2List

def attrVal3AffineMap : AttrVal := [mlir_attr_val| affine_map<(x, y) -> (y)>]
#reduce attrVal3AffineMap

def attrVal4Symbol : AttrVal := [mlir_attr_val| @"foo" ]
#reduce attrVal4Symbol

def attrVal5int: AttrVal := [mlir_attr_val| 42 ]
#reduce attrVal5int

-- def attrVal5bint: AttrVal := [mlir_attr_val| -42 ]
-- #reduce attrVal5bint


def attrVal6Symbol : AttrVal := [mlir_attr_val| @func_foo ]
#reduce attrVal6Symbol

def attrVal7NestedSymbol : AttrVal := [mlir_attr_val| @func_foo::@"func_bar" ]
#reduce attrVal7NestedSymbol


macro_rules
  | `([mlir_attr_val| # $a:str]) =>
      `(AttrVal.alias $a)

def attrVal8Alias : AttrVal := [mlir_attr_val| #"A" ]
#reduce attrVal8Alias


macro_rules
  | `([mlir_attr_val| # $a:ident]) =>
      `(AttrVal.alias $(Lean.quote a.getId.toString))

def attrVal9Alias : AttrVal := [mlir_attr_val| #a ]
#reduce attrVal9Alias

macro_rules
| `([mlir_attr_val|  $x:scientific ]) => `(AttrVal.float $x (MLIRTy.float 64))
| `([mlir_attr_val| $x:scientific : $t:mlir_type]) => `(AttrVal.float $x [mlir_type| $t])


-- def attrVal10Float : AttrVal := [mlir_attr_val| 0.000000e+00  ]
def attrVal10Float :  AttrVal := [mlir_attr_val| 0.0023 ]
#print attrVal10Float

def attrVal11Escape :  AttrVal := [mlir_attr_val| $(attrVal10Float) ]
#print attrVal11Escape

def attrVal12DenseEmpty:  AttrVal := [mlir_attr_val| dense<> : tensor<0 × i64>]
#print attrVal12DenseEmpty


-- MLIR ATTRIBUTE
-- ===============


declare_syntax_cat mlir_attr_entry

syntax ident "=" mlir_attr_val : mlir_attr_entry
syntax strLit "=" mlir_attr_val : mlir_attr_entry
syntax ident : mlir_attr_entry

syntax "[mlir_attr_entry|" mlir_attr_entry "]" : term

-- | TODO: don't actually write an elaborator for the `ident` case. This forces
-- us to declare predefined identifiers in a controlled fashion.
macro_rules 
  | `([mlir_attr_entry| $name:ident  = $v:mlir_attr_val]) => 
     `(AttrEntry.mk $(Lean.quote (name.getId.toString))  [mlir_attr_val| $v])
  | `([mlir_attr_entry| $name:str  = $v:mlir_attr_val]) => 
     `(AttrEntry.mk $name [mlir_attr_val| $v])

macro_rules
  | `([mlir_attr_entry| $name:ident]) =>
     `(AttrEntry.mk $(Lean.quote (name.getId.toString))  AttrVal.unit)

  

def attr0Str : AttrEntry := [mlir_attr_entry| sym_name = "add"]
#print attr0Str

def attr1Type : AttrEntry := [mlir_attr_entry| type = (i32, i32) -> i32]
#print attr1Type

def attr2Escape : AttrEntry :=
   let x : AttrVal := [mlir_attr_val| 42]
   [mlir_attr_entry| sym_name = $(x)]
#print attr0Str


def attr3Unit : AttrEntry :=
   [mlir_attr_entry| sym_name]
#print attr3Unit


declare_syntax_cat mlir_attr_dict
syntax "{" sepBy(mlir_attr_entry, ",") "}" : mlir_attr_dict
syntax "[mlir_attr_dict|" mlir_attr_dict "]" : term

macro_rules
| `([mlir_attr_dict| {  $attrEntries,* } ]) => do
        let attrsList <- attrEntries.getElems.toList.mapM (fun x => `([mlir_attr_entry| $x]))
        let attrsList <- quoteMList attrsList (<- `(MLIR.AST.AttrEntry))
        `(AttrDict.mk $attrsList)


def attrDict0 : AttrDict := [mlir_attr_dict| {}]
def attrDict1 : AttrDict := [mlir_attr_dict| {foo = "bar" }]
def attrDict2 : AttrDict := [mlir_attr_dict| {foo = "bar", baz = "quux" }]

-- dict attribute val
syntax mlir_attr_dict : mlir_attr_val

macro_rules
| `([mlir_attr_val| $v:mlir_attr_dict]) => `(AttrVal.dict [mlir_attr_dict| $v])

def nestedAttrDict0 := [mlir_attr_dict| {foo = {bar = "baz"} }]
#print nestedAttrDict0

-- MLIR OPS WITH REGIONS AND ATTRIBUTES AND BASIC BLOCK ARGS
-- =========================================================


syntax strLit "(" mlir_op_operand,* ")" 
  ("[" mlir_op_successor_arg,* "]")? ("(" mlir_region,* ")")?  (mlir_attr_dict)? ":" mlir_type : mlir_op


macro_rules 
  | `([mlir_op| $$($x) ]) => return x

macro_rules 
  | `([mlir_op| $name:str 
        ( $operands,* )
        $[ [ $succ,* ] ]?
        $[ ( $rgns,* ) ]?
        $[ $attrDict ]? : $ty:mlir_type ]) => do
        let operandsList <- operands.getElems.mapM (fun x => `([mlir_op_operand| $x]))
        let operandsList <- quoteMList operandsList.toList (<- `(MLIR.AST.SSAVal))

        let succList <- match succ with
                | none => `(@List.nil MLIR.AST.BBName)
                | some xs => do 
                  let xs <- xs.getElems.mapM (fun x => `([mlir_op_successor_arg| $x]))
                  quoteMList xs.toList (<- `(MLIR.AST.BBName))
        let attrDict <- match attrDict with 
                          | none => `(AttrDict.mk []) 
                          | some dict => `([mlir_attr_dict| $dict])
        let rgnsList <- match rgns with 
                  | none => `(@List.nil MLIR.AST.Region) 
                  | some rgns => do 
                    let rngs <- rgns.getElems.mapM (fun x => `([mlir_region| $x]))
                    quoteMList rngs.toList (<- `(MLIR.AST.Region))

        `(Op.mk $name -- name
                $operandsList -- operands
                $succList -- bbs
                $rgnsList -- regions
                $attrDict -- attrs
                [mlir_type| $ty]) -- type




def bbstmt1 : BasicBlockStmt := 
  [mlir_bb_stmt| "foo"(%x, %y) : (i32, i32) -> i32]
#print bbstmt1
def bbstmt2: BasicBlockStmt := 
  [mlir_bb_stmt| %z = "foo"(%x, %y) : (i32, i32) -> i32]
#print bbstmt2

def bbop1 : SSAVal × MLIRTy := [mlir_bb_operand| %x : i32 ]
#print bbop1

def bb1NoArgs : BasicBlock := 
  [mlir_bb|
     ^entry:
     "foo"(%x, %y) : (i32, i32) -> i32
      %z = "bar"(%x) : (i32) -> (i32)
      "std.return"(%x0) : (i42) -> ()
  ]
#print bb1NoArgs

def bb2SingleArg : BasicBlock := 
  [mlir_bb|
     ^entry(%argp : i32):
     "foo"(%x, %y) : (i32, i32) -> i32
      %z = "bar"(%x) : (i32) -> (i32)
      "std.return"(%x0) : (i42) -> ()
  ]
#print bb2SingleArg


def bb3MultipleArgs : BasicBlock := 
  [mlir_bb|
     ^entry(%argp : i32, %argq : i64):
     "foo"(%x, %y) : (i32, i32) -> i32
      %z = "bar"(%x) : (i32) -> (i32)
      "std.return"(%x0) : (i42) -> ()
  ]
#reduce bb3MultipleArgs


def rgn0 : Region := ([mlir_region|  { }])
#print rgn0

def rgn1 : Region := 
  [mlir_region|  { 
    ^entry:
      "std.return"(%x0) : (i42) -> ()
  }]
#print rgn1

def rgn2 : Region := 
  [mlir_region|  { 
    ^entry:
      "std.return"(%x0) : (i42) -> ()

    ^loop:
      "std.return"(%x1) : (i42) -> ()
  }]
#print rgn2

-- | test what happens if we try to use an entry block with no explicit bb name
def rgn3 : Region := 
  [mlir_region|  { 
      "std.return"(%x0) : (i42) -> ()
  }]
#print rgn1


-- | test simple ops [no regions]
def opcall1 : Op := [mlir_op| "foo" (%x, %y) : (i32, i32) -> i32 ]
#print opcall1


def opattr0 : Op := [mlir_op|
 "foo"() { sym_name = "add", type = (i32, i32) -> i32 } : () -> ()
]
#print opattr0


def oprgn0 : Op := [mlir_op|
 "func"() ( {
  ^bb0(%arg0: i32, %arg1: i32):
    %x = "std.addi"(%arg0, %arg1) : (i32, i32) -> i32
    "std.return"(%x) : (i32) -> ()
  }) : () -> ()
]
#print oprgn0


-- | note that this is a "full stack" example!
def opRgnAttr0 : Op := [mlir_op|
 "module"() (
 {
  ^entry:
   "func"() (
    {
     ^bb0(%arg0:i32, %arg1:i32):
      %zero = "std.addi"(%arg0 , %arg1) : (i32, i32) -> i32
      "std.return"(%zero) : (i32) -> ()
    }){sym_name = "add", type = (i32, i32) -> i32} : () -> ()
   "module_terminator"() : () -> ()
 }) : () -> ()
]
#print opRgnAttr0



-- | test simple ops [no regions, but with bb args]
def opcall2 : Op := [mlir_op| "foo" (%x, %y) [^bb1, ^bb2] : (i32, i32) -> i32]
#print opcall2


-- | Builtins
-- =========

syntax "func" mlir_attr_val_symbol "(" sepBy(mlir_bb_operand, ",") ")" mlir_region : mlir_op

-- | note that this only supports single BB region.
-- | TODO: add support for multi BB region.
macro_rules 
| `([mlir_op| func $name:mlir_attr_val_symbol ( $args,* )  $rgn ]) => do
     let argsList <- args.getElems.mapM (fun x => `([mlir_bb_operand| $x])) 
     let argsList <- quoteMList argsList.toList (<- `(MLIR.AST.SSAVal × MLIR.AST.MLIRTy))

     let typesList <- args.getElems.mapM (fun x => `(Prod.snd [mlir_bb_operand| $x])) 
     let typesList <- quoteMList typesList.toList (<- `(MLIR.AST.MLIRTy))

     let rgn <- `([mlir_region| $rgn])
     let rgn <- `(($rgn).ensureEntryBlock.setEntryBlockArgs $argsList)
     let argTys <- `(($argsList).map Prod.snd)
     let ty <-  `(MLIRTy.fn (MLIRTy.tuple $typesList) MLIRTy.unit)
     let attrs <- `(AttrDict.empty.addType "type" $ty)
     let attrs <- `(($attrs).add ("sym_name", [mlir_attr_val_symbol| $name]))

     `(Op.mk "func" [] [] [$rgn] $attrs $ty)


def func1 : Op := [mlir_op| func @"main"() {
  ^entry:
    %x = "asm.int" () { "val" = 32 } : () -> (i32)
}]

syntax "module" "{" mlir_op* "}" : mlir_op

macro_rules 
| `([mlir_op| module { $ops* } ]) => do
     let initList <- `(@List.nil MLIR.AST.Op)
     let ops <- ops.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_op| $x] ])
     let ops <- `($ops ++ [Op.empty "module_terminator"])
     let rgn <- `(Region.fromOps $ops)
     `(Op.mk "module" [] [] [$rgn] AttrDict.empty [mlir_type| () -> ()])

def mod1 : Op := [mlir_op| module { }]
#print mod1

--- MEMREF+TENSOR
--- =============
-- dimension-list ::= dimension-list-ranked | (`*` `x`)
-- dimension-list-ranked ::= (dimension `x`)*
-- dimension ::= `?` | decimal-literal
-- tensor-type ::= `tensor` `<` dimension-list tensor-memref-element-type `>`
-- tensor-memref-element-type ::= vector-element-type | vector-type | complex-type


-- https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype
-- memref-type ::= ranked-memref-type | unranked-memref-type
-- ranked-memref-type ::= `memref` `<` dimension-list-ranked type
--                        (`,` layout-specification)? (`,` memory-space)? `>`
-- unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`
-- stride-list ::= `[` (dimension (`,` dimension)*)? `]`
-- strided-layout ::= `offset:` dimension `,` `strides: ` stride-list
-- layout-specification ::= semi-affine-map | strided-layout | attribute-value
-- memory-space ::= attribute-value
-- | good example for paper.
declare_syntax_cat memref_type_stride_list
syntax "[" (mlir_dimension,*) "]" : memref_type_stride_list

declare_syntax_cat memref_type_strided_layout
syntax "offset:" mlir_dimension "," "strides:" memref_type_stride_list : memref_type_strided_layout

declare_syntax_cat memref_type_layout_specification
syntax memref_type_strided_layout : memref_type_layout_specification
syntax mlir_attr_val : memref_type_layout_specification
syntax "[memref_type_layout_specification|" memref_type_layout_specification "]" : term


macro_rules
| `([memref_type_layout_specification| $v:mlir_attr_val]) => 
    `(MemrefLayoutSpec.attr [mlir_attr_val| $v])
| `([memref_type_layout_specification| offset: $o:mlir_dimension , strides: [ $[ $ds:mlir_dimension ],* ]]) =>  do
    let ds <- ds.mapM (fun d => `([mlir_dimension| $d]))
    let ds <- quoteMList ds.toList (<- `(MLIR.AST.Dimension))
    `(MemrefLayoutSpec.stride [mlir_dimension| $o] $ds)

-- | ranked memref
syntax "memref" "<"  mlir_dimension_list ("," memref_type_layout_specification)? ("," mlir_attr_val)?  ">"  : mlir_type
macro_rules
| `([mlir_type| memref  < $dims:mlir_dimension_list $[, $layout ]? $[, $memspace]? >]) => do
    let (dims, ty) <- parseTensorDimensionList dims 
    let memspace <- match memspace with 
                    | some s => `(some [mlir_attr_val| $s])
                    | none => `(none)

    let layout <- match layout with 
                  | some stx => `(some [memref_type_layout_specification| $stx])
                  | none => `(none)
    `(MLIRTy.memrefRanked $dims $ty $layout $memspace)

def memrefTy0 := [mlir_type| memref<3×3×i32>]
#print memrefTy0

def memrefTy1 := [mlir_type| memref<i32>]
#print memrefTy1

def memrefTy2 := [mlir_type| memref<2 × 4 × i8, #map1>]
#print memrefTy2

def memrefTy3 := [mlir_type| memref<2 × 4 × i8, #map1, 1>]
#print memrefTy3
  

-- | unranked memref
-- unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`
-- | TODO: Do I need two different parsers for these cases?
syntax "memref" "<"  "*" "×" mlir_type ("," mlir_attr_val)?  ">"  : mlir_type
syntax "memref" "<*" "×" mlir_type ("," mlir_attr_val)?  ">"  : mlir_type
macro_rules
| `([mlir_type| memref < * × $ty  $[, $memspace]? >]) => do
    let memspace <- match memspace with 
                    | some s => `(some [mlir_attr_val| $s])
                    | none => `(none)
    `(MLIRTy.memrefUnranked [mlir_type| $ty] $memspace)

macro_rules
| `([mlir_type| memref <* × $ty  $[, $memspace]? >]) => do
    let memspace <- match memspace with 
                    | some s => `(some [mlir_attr_val| $s])
                    | none => `(none)
    `(MLIRTy.memrefUnranked [mlir_type| $ty] $memspace)
 
def memrefTy4 := [mlir_type| memref<* × f32>]
#print memrefTy4




-- | FML, we need to manually split on the 'dot'
/-
macro "[mlir_type|" "!" dialectAndName:ident  b:balancedBrackets "]" : term => do
   let dialectAndNameStr := dialectAndName.getId.toString
   let splits := dialectAndNameStr.splitOn "."
   let dialectStr := splits.get! 0
   let nameStr := splits.get! 1
   -- dbg_trace "b: {b}"
   let bStr := match b[0] with | .atom _ val => val | _ => panic! "expected balanced brackets to have atom"
   let t := MLIRTy.userPretty dialectStr nameStr bStr
   `(MLIRTy.userPretty $(Lean.quote dialectStr)
                       $(Lean.quote nameStr)
                       $(Lean.quote bStr))
-/

syntax "!" ident balancedBrackets : mlir_type
macro_rules
| `([mlir_type| ! $dialectAndName  $b]) => do
   let dialectAndNameStr := dialectAndName.getId.toString
   let splits := dialectAndNameStr.splitOn "."
   let dialectStr := splits.get! 0
   let nameStr := splits.get! 1
   -- dbg_trace "b: {b}"
   let bStr := match b[0] with | .atom _ val => val | _ => panic! "expected balanced brackets to have atom"
   let t := MLIRTy.userPretty dialectStr nameStr bStr
   `(MLIRTy.userPretty $(Lean.quote dialectStr)
                       $(Lean.quote nameStr)
                       $(Lean.quote bStr))

set_option pp.rawOnError true
def userTy0 := [mlir_type| !test.test_rec<a, test_rec<b, test_type>> ]
#print userTy0

-- | TODO: how do we make this work?
def userTy1 := [mlir_type| () -> !foo.bar<foo> ]
#print userTy1

-- | TODO: how do we make this work?
def userTy2 := [mlir_type| !foo.bar<foo> -> ()]
#print userTy2


end MLIR.EDSL

