import MLIR.AST
import Lean.Parser
import Lean.Parser.Extra
-- import Lean.Init.Meta


open Lean
open Lean.Parser

open MLIR.AST

namespace MLIR.EDSL
-- EDSL
-- ====

declare_syntax_cat mlir_bb
declare_syntax_cat mlir_region
declare_syntax_cat mlir_bb_stmt
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
syntax "<[" term "]>" : mlir_op_operand
syntax "{{" term "}}" : mlir_op_operand

syntax "[mlir_op_operand| " mlir_op_operand "]" : term -- translate operands into term
macro_rules
  | `([mlir_op_operand| % $x:ident]) => `(SSAVal.SSAVal $(Lean.quote (toString x.getId))) 
  | `([mlir_op_operand| <[ $t:term ]> ]) => t
  | `([mlir_op_operand| {{ $t:term }} ]) => t

def xx := ([mlir_op_operand| %x])
def xxx := ([mlir_op_operand| %x])
#print xx
#print xxx


-- EDSL OP-CALL-ARGS
-- =================

syntax "(" ")" : mlir_op_args
syntax "(" mlir_op_operand ")" : mlir_op_args
syntax "(" mlir_op_operand "," mlir_op_operand","* ")" : mlir_op_args

syntax "mlir_op_args% " mlir_op_args : term -- translate mlir_op args into term
macro_rules
  | `(mlir_op_args% ( ) ) => `([])
  | `(mlir_op_args% ( $x:mlir_op_operand ) ) => 
      `([ [mlir_op_operand| $x] ])
  | `(mlir_op_args% ( $x:mlir_op_operand, $y:mlir_op_operand ) ) => 
      `([[mlir_op_operand| $x], [mlir_op_operand| $y]])


def call0 : List SSAVal := (mlir_op_args% ())
def call1 : List SSAVal := (mlir_op_args% (%x))
def call2 : List SSAVal := (mlir_op_args% (%x, %y))
#print call0
#print call1
#print call2


-- EDSL OP-SUCCESSOR-ARGS
-- =================

-- successor-list       ::= `[` successor (`,` successor)* `]`
-- successor            ::= caret-id (`:` bb-arg-list)?

declare_syntax_cat mlir_op_successor_arg -- bb argument
syntax "^" ident : mlir_op_successor_arg -- bb argument with no operands
-- syntax "^" ident ":" "(" mlir_op_operand","* ")" : mlir_op_successor_arg

syntax "mlir_op_successor_arg% " mlir_op_successor_arg : term

macro_rules
  | `(mlir_op_successor_arg% ^ $x:ident  ) => 
      `(BBName.mk $(Lean.quote (toString x.getId)))

def succ0 :  BBName := (mlir_op_successor_arg% ^bb)
#print succ0


-- EDSL MLIR TYPES
-- ===============

syntax "(" ")" : mlir_type
syntax "(" mlir_type ")" : mlir_type
syntax "(" mlir_type "," mlir_type ")" : mlir_type
syntax mlir_type "->" mlir_type : mlir_type
syntax ident: mlir_type

-- | TODO: fix this rule, it interfers with way too much other stuff!
-- syntax "i" numLit : mlir_type

syntax "mlir_type%" mlir_type : term

macro_rules
  | `(mlir_type% $x:ident ) => do
        let xstr := x.getId.toString
        if xstr.front == 'i'
        then do 
          let xstr' := xstr.drop 1
          let lit := Lean.Syntax.mkNumLit xstr'
          `(MLIRTy.int $lit)
        else Macro.throwError "expected i<int>" -- `(MLIRTy.int 1337)

def tyi32NoGap : MLIRTy := (mlir_type% i32) -- TODO: how to keep no gap?

macro_rules
  | `(mlir_type% ( ) ) => `(MLIRTy.tuple [])
  | `(mlir_type% ( $x:mlir_type ) ) => `(MLIRTy.tuple [(mlir_type% $x)])
  | `(mlir_type% ( $x:mlir_type, $y:mlir_type ) ) => `(MLIRTy.tuple [(mlir_type% $x), (mlir_type% $y)])
  -- | `(mlir_type% i $x:numLit ) => `(MLIRTy.int $x)
  | `(mlir_type% $dom:mlir_type -> $codom:mlir_type) => `(MLIRTy.fn (mlir_type% $dom) (mlir_type% $codom))

def ty0 : MLIRTy := (mlir_type% ())
def tyi32 : MLIRTy := (mlir_type% i32) -- TODO: how to keep no gap?
-- def tyi32' : MLIRTy := (mlir_type% i32) -- TODO: how to keep no gap?
def tysingle : MLIRTy := (mlir_type% (i42))
def typair : MLIRTy := (mlir_type% (i32, i64))
def tyfn0 : MLIRTy := (mlir_type% () -> ())
def tyfn1 : MLIRTy := (mlir_type% (i11) -> (i12))
def tyfn2 : MLIRTy := (mlir_type% (i21, i22) -> (i23, i24))
#print ty0
#print tyi32
#print typair
#print tyfn0
#print tyfn1
-- #print tyi32'


-- EDSL MLIR OP CALL, MLIR BB STMT
-- ===============================

-- syntax strLit mlir_op_args ":" mlir_type : mlir_op

syntax "[mlir_op|" mlir_op "]" : term


syntax mlir_op: mlir_bb_stmt
syntax mlir_op_operand "=" mlir_op : mlir_bb_stmt
syntax "mlir_bb_stmt%" mlir_bb_stmt : term


macro_rules
  | `(mlir_bb_stmt% $call:mlir_op ) =>
       `(BasicBlockStmt.StmtOp ([mlir_op| $call]))
  | `(mlir_bb_stmt% $res:mlir_op_operand = $call:mlir_op) => 
       `(BasicBlockStmt.StmtAssign ([mlir_op_operand| $res]) ([mlir_op| $call]))




-- EDSL MLIR BASIC BLOCK OPERANDS
-- ==============================

declare_syntax_cat mlir_bb_operand
syntax mlir_op_operand ":" mlir_type : mlir_bb_operand

syntax "mlir_bb_operand%" mlir_bb_operand : term

macro_rules 
| `(mlir_bb_operand% $name:mlir_op_operand : $ty:mlir_type ) => 
     `( ([mlir_op_operand| $name], mlir_type% $ty) ) 



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


-- EDSL MLIR REGIONS
-- =================

syntax "{" (ws mlir_bb ws)* "}": mlir_region
syntax "mlir_region% " mlir_region : term
syntax "<[" term "]>" : mlir_region

macro_rules
| `(mlir_region% { $[ $bbs ]* }) => do
   let initList <- `([])
   let bbsList <- bbs.foldlM (init := initList) fun xs x => `($xs ++ [mlir_bb% $x])
   `(Region.mk $bbsList)

macro_rules
| `(mlir_region% <[ $t: term ]>) => t



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

def attrVal1Ty : AttrVal := mlir_attr_val% (i32, i64) -> i32
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

def attr1Type : Attr := (mlir_attr% type = (i32, i32) -> i32)
#print attr1Type

-- MLIR OPS WITH REGIONS AND ATTRIBUTES AND BASIC BLOCK ARGS
-- =========================================================


syntax strLit mlir_op_args ("[" mlir_op_successor_arg,* "]")? ("(" mlir_region,* ")")?  ("{" mlir_attr,* "}")? ":" mlir_type : mlir_op


macro_rules 
  | `([mlir_op| $name:strLit $args:mlir_op_args
        $[ [ $succ,* ] ]?
        $[ ( $rgns,* ) ]?
        $[ { $attrs,* } ]? : $ty:mlir_type ]) => do
        let initList <- `([])
        let succList <- match succ with
                | none => `([])
                | some xs => xs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [mlir_op_successor_arg% $x])
        let attrsList <- match attrs with 
                          | none => `([]) 
                          | some attrs => attrs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [mlir_attr% $x])
        let rgnsList <- match rgns with 
                          | none => `([]) 
                          | some rgns => rgns.getElems.foldlM (init := initList) fun xs x => `($xs ++ [mlir_region% $x])
        `(Op.mk $name -- name
                (mlir_op_args% $args) -- args
                $succList -- bbs
                $rgnsList -- regions
                $attrsList -- attrs
                (mlir_type% $ty)) -- type



def bbstmt1 : BasicBlockStmt := (mlir_bb_stmt% "foo"(%x, %y) : (i32, i32) -> i32)
#print bbstmt1
def bbstmt2: BasicBlockStmt := (mlir_bb_stmt% %z = "foo"(%x, %y) : (i32, i32) -> i32)
#print bbstmt2

def bbop1 : SSAVal × MLIRTy := mlir_bb_operand% %x : i32
#print bbop1

def bb1NoArgs : BasicBlock := 
  (mlir_bb%
     ^entry:
     "foo"(%x, %y) : (i32, i32) -> i32
      %z = "bar"(%x) : (i32) -> (i32)
      "std.return"(%x0) : (i42) -> ()

  )
#print bb1NoArgs

def bb2SingleArg : BasicBlock := 
  (mlir_bb%
     ^entry(%argp : i32):
     "foo"(%x, %y) : (i32, i32) -> i32
      %z = "bar"(%x) : (i32) -> (i32)
      "std.return"(%x0) : (i42) -> ()

  )
#print bb2SingleArg


def bb3MultipleArgs : BasicBlock := 
  (mlir_bb%
     ^entry(%argp : i32, %argq : i64):
     "foo"(%x, %y) : (i32, i32) -> i32
      %z = "bar"(%x) : (i32) -> (i32)
      "std.return"(%x0) : (i42) -> ()

  )
#print bb3MultipleArgs


def rgn0 : Region := (mlir_region%  { })
#print rgn0

def rgn1 : Region := 
  (mlir_region%  { 
    ^entry:
      "std.return"(%x0) : (i42) -> ()
  })
#print rgn1

def rgn2 : Region := 
  (mlir_region%  { 
    ^entry:
      "std.return"(%x0) : (i42) -> ()

    ^loop:
      "std.return"(%x1) : (i42) -> ()
  })
#print rgn2


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

end MLIR.EDSL
