import MLIR.AST
import Lean.Parser
import Lean.Parser.Extra
-- import Lean.Init.Meta


open Lean
open Lean.Parser

open MLIR.AST

namespace MLIR.EDSL

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
   let initList  <- `([])
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

syntax "%" ident : mlir_op_operand
syntax "[escape|" term "]" : mlir_op_operand

syntax "[mlir_op_operand| " mlir_op_operand "]" : term -- translate operands into term
macro_rules
  | `([mlir_op_operand| % $x:ident]) => `(SSAVal.SSAVal $(Lean.quote (toString x.getId))) 
  | `([mlir_op_operand| [escape| $t:term ] ]) => return t

def operand0 := [mlir_op_operand| %x]
#print operand0

def operand1 := [mlir_op_operand| %x]
#print operand1


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
      `(BBName.mk $(Lean.quote (toString x.getId)))

def succ0 :  BBName := ([mlir_op_successor_arg| ^bb])
#print succ0


-- EDSL MLIR TYPES
-- ===============

syntax "(" ")" : mlir_type
syntax "(" mlir_type ")" : mlir_type
syntax "(" mlir_type "," mlir_type ")" : mlir_type
syntax mlir_type "->" mlir_type : mlir_type
syntax "{{" term "}}" : mlir_type
syntax "!" str : mlir_type
syntax ident: mlir_type

-- | TODO: fix this rule, it interfers with way too much other stuff!
-- syntax "i" numLit : mlir_type

syntax "[mlir_type|" mlir_type "]" : term


set_option hygiene false in -- allow i to expand 
macro_rules
  | `([mlir_type| $x:ident ]) => do
        let xstr := x.getId.toString
        if xstr.front == 'i' || xstr.front == 'f'
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
| `([mlir_type| ! $x ]) => `(MLIRTy.user $x)
def tyUser : MLIRTy := [mlir_type| !"lz.int"]
#eval tyUser

def tyi32NoGap : MLIRTy := [mlir_type| i32]
#eval tyi32NoGap
def tyf32NoGap : MLIRTy := [mlir_type| f32]
#eval tyf32NoGap

macro_rules
| `([mlir_type| {{ $t }} ]) => return t -- antiquot type

macro_rules
  | `([mlir_type| ( ) ]) => `(MLIRTy.tuple [])
  | `([mlir_type| ( $x:mlir_type )]) => 
        `(MLIRTy.tuple [ [mlir_type|$x] ])
  | `([mlir_type| ( $x:mlir_type, $y:mlir_type )]) => 
    `(MLIRTy.tuple [ [mlir_type|$x], [mlir_type|$y] ] )
  -- | `([mlir_type| i $x:numLit ) => `(MLIRTy.int $x)
  | `([mlir_type| $dom:mlir_type -> $codom:mlir_type]) =>
     `(MLIRTy.fn [mlir_type|$dom] [mlir_type|$codom])

def ty0 : MLIRTy := [mlir_type| ()]
def tyi32 : MLIRTy := [mlir_type| i32] -- TODO: how to keep no gap?
-- def tyi32' : MLIRTy := ([mlir_type| i32) -- TODO: how to keep no gap?
def tysingle : MLIRTy := [mlir_type| (i42)]
def typair : MLIRTy := [mlir_type| (i32, i64)]
def tyfn0 : MLIRTy := [mlir_type| () -> ()]
def tyfn1 : MLIRTy := [mlir_type| (i11) -> (i12)]
def tyfn2 : MLIRTy := [mlir_type| (i21, i22) -> (i23, i24)]
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
| `([mlir_dimension| $x:numLit ]) => 
    `(Dimension.Known $x)

def dim0 := [mlir_dimension| 30]
#print dim0

def dim1 := [mlir_dimension| ?]
#print dim1


-- TODO: where is vector type syntax defined?
-- | TODO: fix bug that does not allow a trailing times.
-- The grammar should be: 
-- syntax "vector" "<" sepBy1(mlir_dimension, "×") "×" mlir_type ">"  : mlir_type
syntax "vector" "<" sepBy1(mlir_dimension, "×") ":" mlir_type ">"  : mlir_type
macro_rules
| `([mlir_type| vector < $[ $dims ]×* : $ty:mlir_type  >]) => do
    let initList <- `([])
    let dimsList <- dims.foldlM (init := initList) fun ds d => `($ds ++ [[mlir_dimension| $d]])
    `(MLIRTy.vector $dimsList [mlir_type| $ty])


-- | TODO: fix bug that does not allow a trailing times.

syntax "tensor" "<" sepBy1(mlir_dimension, "×") ":" mlir_type ">"  : mlir_type
macro_rules
| `([mlir_type| tensor < $[ $dims ]×* : $ty:mlir_type  >]) => do
    let initList <- `([])
    let dimsList <- dims.foldlM (init := initList) fun ds d => `($ds ++ [[mlir_dimension| $d]])
    `(MLIRTy.tensor $dimsList [mlir_type| $ty])


def tensorTy0 := [mlir_type| tensor<3×3:i32>]
#print tensorTy0
def tensorTy1 := [mlir_type| tensor<3×?:f32>]
#print tensorTy1
     
      

-- EDSL MLIR OP CALL, MLIR BB STMT
-- ===============================

-- syntax strLit mlir_op_args ":" mlir_type : mlir_op

syntax "[mlir_op|" mlir_op "]" : term



syntax mlir_op: mlir_bb_stmt
syntax mlir_op_operand "=" mlir_op : mlir_bb_stmt
syntax "{{" term "}}" : mlir_bb_stmt

syntax "[mlir_bb_stmt|" mlir_bb_stmt "]" : term

syntax "[escape|" term "]" : mlir_bb_stmt

macro_rules
  | `([mlir_bb_stmt| $call:mlir_op ]) =>
       `(BasicBlockStmt.StmtOp ([mlir_op| $call]))
  | `([mlir_bb_stmt| $res:mlir_op_operand = $call:mlir_op]) => 
       `(BasicBlockStmt.StmtAssign ([mlir_op_operand| $res]) ([mlir_op| $call]))
  | `([mlir_bb_stmt| {{ $t }} ]) => return t

macro_rules
| `([mlir_bb_stmt| [escape| $t ]]) => `(coe $t)



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

syntax (ws mlir_bb_stmt ws)* : mlir_bb_stmts

syntax "[mlir_bb_stmts|" mlir_bb_stmts "]" : term
macro_rules
| `([mlir_bb_stmts| $[ $stmts ]*  ]) => do
      let initList <- `([])
      stmts.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_bb_stmt|$x]])


syntax "[escape|" term "]" : mlir_bb_stmts
macro_rules 
| `([mlir_bb_stmts| [escape| $t ] ]) => return t


syntax "[mlir_bb|" mlir_bb "]": term


macro_rules 
| `([mlir_bb| ^ $name:ident ( $operands,* ) : $stmts ]) => do
   let initList <- `([])
   let argsList <- operands.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_bb_operand| $x]])
   let opsList <- `([mlir_bb_stmts| $stmts])
   `(BasicBlock.mk $(Lean.quote (toString name.getId)) $argsList $opsList)
| `([mlir_bb| ^ $name:ident : $stmts ]) => do
   let initList <- `([])
   let opsList <- `([mlir_bb_stmts| $stmts])
   `(BasicBlock.mk $(Lean.quote (toString name.getId)) [] $opsList)


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
syntax "[escape|" term "]" : mlir_region

-- | map a macro on a list

macro_rules
| `([mlir_region| { $[ $entrybb ]? $[ $bbs ]* } ]) => do
   let initList <- match entrybb with 
                  | some entry => `([[mlir_entry_bb| $entry]])
                  | none => `([])
   let bbsList <- bbs.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_bb|$x]])
   `(Region.mk $bbsList)

macro_rules
| `([mlir_region| [escape| $t: term ] ]) => return t


-- TENSOR LITERAL
-- ==============

declare_syntax_cat mlir_tensor
syntax numLit : mlir_tensor
syntax "[" sepBy(mlir_tensor, ",") "]" : mlir_tensor

syntax "[mlir_tensor|" mlir_tensor "]" : term

macro_rules
| `([mlir_tensor| $x:numLit ]) => `(TensorElem.int $x)

macro_rules
| `([mlir_tensor| [ $xs,* ] ]) => do 
    let initList <- `([])
    let vals <- xs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_tensor| $x]]) 
    `(TensorElem.nested $vals)


-- MLIR ATTRIBUTE VALUE
-- ====================

-- | TODO: consider renaming this to mlir_attr
declare_syntax_cat mlir_attr_val
declare_syntax_cat mlir_attr_val_symbol
syntax "@" str : mlir_attr_val_symbol


syntax str: mlir_attr_val
syntax mlir_type : mlir_attr_val
syntax affine_map : mlir_attr_val
syntax mlir_attr_val_symbol : mlir_attr_val
syntax num (":" mlir_type)? : mlir_attr_val


syntax "[" sepBy(mlir_attr_val, ",") "]" : mlir_attr_val
syntax "[escape|" term "]" : mlir_attr_val
syntax "[mlir_attr_val|" mlir_attr_val "]" : term
syntax "[mlir_attr|" mlir_attr_val "]" : term
macro_rules
| `([mlir_attr|  $x ]) => `([mlir_attr_val| $x ])

macro_rules
| `([mlir_attr_val| [escape| $x:term ] ]) => `($x)

macro_rules
| `([mlir_attr_val|  $x:numLit ]) => `(AttrVal.int $x (MLIRTy.int 64))
| `([mlir_attr_val| $x:numLit : $t:mlir_type]) => `(AttrVal.int $x [mlir_type| $t])
-- | TODO: how to get mlir_type?
-- macro_rules



macro_rules 
  | `([mlir_attr_val| $s:strLit]) => `(AttrVal.str $s)
  | `([mlir_attr_val| [ $xs,* ] ]) => do 
        let initList <- `([])
        let vals <- xs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_attr_val| $x]]) 
        `(AttrVal.list $vals)
  | `([mlir_attr_val| $ty:mlir_type]) => `(AttrVal.type [mlir_type| $ty])


syntax "dense<" mlir_tensor  ">" ":" mlir_type : mlir_attr_val
macro_rules
| `([mlir_attr_val| dense< $v:mlir_tensor > : $t:mlir_type]) => 
    `(AttrVal.dense [mlir_tensor| $v] [mlir_type| $t])

macro_rules
  | `([mlir_attr_val| $a:affine_map]) =>
      `(AttrVal.affine [affine_map| $a])

macro_rules
| `([mlir_attr_val| @ $x:strLit ]) =>
      `(AttrVal.symbol $x)

def attrVal0Str : AttrVal := [mlir_attr_val| "foo"]
#reduce attrVal0Str

def attrVal1Ty : AttrVal := [mlir_attr_val| (i32, i64) -> i32]
#reduce attrVal1Ty

def attrVal2List : AttrVal := [mlir_attr_val| ["foo", "foo"] ]
#reduce attrVal2List

def attrVal3AffineMap : AttrVal := [mlir_attr_val| affine_map<(x, y) -> (y)>]
#reduce attrVal3AffineMap

def attrVal4Symbol : AttrVal := [mlir_attr_val| @"foo" ]
#reduce attrVal4Symbol

def attrVal5int: AttrVal := [mlir_attr_val| 42 ]
#reduce attrVal5int



-- MLIR ATTRIBUTE
-- ===============


declare_syntax_cat mlir_attr_entry

syntax ident "=" mlir_attr_val : mlir_attr_entry
syntax strLit "=" mlir_attr_val : mlir_attr_entry

syntax "[mlir_attr_entry|" mlir_attr_entry "]" : term

macro_rules 
  | `([mlir_attr_entry| $name:ident  = $v:mlir_attr_val]) => 
     `(AttrEntry.mk $(Lean.quote (toString name.getId))  [mlir_attr_val| $v])
  | `([mlir_attr_entry| $name:strLit  = $v:mlir_attr_val]) => 
     `(AttrEntry.mk $name [mlir_attr_val| $v])

def attr0Str : AttrEntry := [mlir_attr_entry| sym_name = "add"]
#print attr0Str

def attr1Type : AttrEntry := [mlir_attr_entry| type = (i32, i32) -> i32]
#print attr1Type


declare_syntax_cat mlir_attr_dict
syntax "{" sepBy(mlir_attr_entry, ",") "}" : mlir_attr_dict
syntax "[mlir_attr_dict|" mlir_attr_dict "]" : term

macro_rules
| `([mlir_attr_dict| {  $attrEntries,* } ]) => do
        let initList <- `([])
        let attrsList <-attrEntries.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_attr_entry| $x]]) 
        `(AttrDict.mk $attrsList)


def attrDict0 : AttrDict := [mlir_attr_dict| {}]
def attrDict1 : AttrDict := [mlir_attr_dict| {foo = "bar" }]
def attrDict2 : AttrDict := [mlir_attr_dict| {foo = "bar", baz = "quux" }]

-- MLIR OPS WITH REGIONS AND ATTRIBUTES AND BASIC BLOCK ARGS
-- =========================================================


-- | TODO: Replace |{ mlir_attr_entry,*}| with |mlir_attr|.
syntax strLit "(" mlir_op_operand,* ")" 
  ("[" mlir_op_successor_arg,* "]")? ("(" mlir_region,* ")")?  ("{" mlir_attr_entry,* "}")? ":" mlir_type : mlir_op


syntax "[escape|" term "]" : mlir_op
macro_rules 
  | `([mlir_op| [escape| $x ] ]) => return x

macro_rules 
  | `([mlir_op| $name:strLit 
        ( $operands,* )
        $[ [ $succ,* ] ]?
        $[ ( $rgns,* ) ]?
        $[ { $attrs,* } ]? : $ty:mlir_type ]) => do
        let initList <- `([])
        let operandsList <- operands.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_op_operand| $x]])
        let succList <- match succ with
                | none => `([])
                | some xs => xs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_op_successor_arg| $x] ])
        let attrsList <- match attrs with 
                          | none => `([]) 
                          | some attrs => attrs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_attr_entry| $x]])
        let rgnsList <- match rgns with 
                          | none => `([]) 
                          | some rgns => rgns.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_region| $x]])
        `(Op.mk $name -- name
                $operandsList -- operands
                $succList -- bbs
                $rgnsList -- regions
                (AttrDict.mk $attrsList) -- attrs
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
#print bb3MultipleArgs


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
| `([mlir_op| func @$name ( $args,* )  $rgn ]) => do
     let initList <- `([])
     let argsList <- args.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_bb_operand| $x]])
     let typesList <- args.getElems.foldlM (init := initList) fun xs x => `($xs ++ [Prod.snd [mlir_bb_operand| $x]])
     let rgn <- `([mlir_region| $rgn])
     let rgn <- `(($rgn).ensureEntryBlock.setEntryBlockArgs $argsList)
     let argTys <- `(($argsList).map Prod.snd)
     let ty <-  `(MLIRTy.fn (MLIRTy.tuple $typesList) MLIRTy.unit)
     let attrs <- `(AttrDict.empty.addType "type" $ty)
     let attrs <- `(($attrs).addString "sym_name" $name)

     `(Op.mk "func" [] [] [$rgn] $attrs $ty)


def func1 : Op := [mlir_op| func @"main"() {
  ^entry:
    %x = "asm.int" () { "val" = 32 } : () -> (i32)
}]

syntax "module" "{" mlir_op* "}" : mlir_op

macro_rules 
| `([mlir_op| module { $ops* } ]) => do
     let initList <- `([])
     let ops <- ops.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_op| $x] ])
     let ops <- `($ops ++ [Op.empty "module_terminator"])
     let rgn <- `(Region.fromOps $ops)
     `(Op.mk "module" [] [] [$rgn] AttrDict.empty [mlir_type| () -> ()])

def mod1 : Op := [mlir_op| module { }]
#print mod1


end MLIR.EDSL
