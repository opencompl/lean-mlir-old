import MLIR.AST
import Lean.Parser
import Lean.Parser.Extra


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
syntax "i" numLit : mlir_type

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
end MLIR.EDSL
