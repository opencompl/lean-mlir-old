import MLIR.AST
import MLIR.EDSL
import MLIR.Doc

open MLIR.AST
open MLIR.EDSL
open MLIR.Doc
open Std

-- https://mlir.llvm.org/docs/Dialects/Standard/        
-- -- some delaborators: https://github.com/leanprover/lean4/blob/68867d02ac1550288427195fa09e46866bd409b8/src/Init/NotationExtra.lean


syntax "addi" mlir_op_operand "," mlir_op_operand : mlir_op
syntax "addf" mlir_op_operand "," mlir_op_operand ":" mlir_type : mlir_op
syntax "mulf" mlir_op_operand "," mlir_op_operand ":" mlir_type : mlir_op

syntax "br" mlir_op_successor_arg : mlir_op
syntax "cond_br" mlir_op_operand "," mlir_op_successor_arg "," mlir_op_successor_arg : mlir_op

-- | this is a hack, look into using eraseMacroScopes: 
-- > You can use eraseMacroScopes to get the name the user typed (i).
-- > I think you can also match on the i directly (i instead of $x:ident),
-- > but I'm not sure how that interacts with hygiene.
-- https://leanprover.zulipchat.com/#narrow/stream/270676-lean4/topic/Disabling.20Macro.20Hygine.3F/near/256933681
set_option hygiene false in -- need to disable hygiene for i32 expansion.
macro_rules
  | `([mlir_op| addi $op1:mlir_op_operand , $op2:mlir_op_operand]) => 
        `( [mlir_op| "std.addi" ( $op1  , $op2) : (i32, i32) -> (i32) ] )

set_option hygiene false in -- need to disable hygiene for type expansion
macro_rules
  | `([mlir_op| addf $op1:mlir_op_operand , $op2:mlir_op_operand : $ty:mlir_type]) => 
        `( [mlir_op| "std.addf" ($op1, $op2) : ($ty, $ty) -> ($ty) ] )

set_option hygiene false in -- need to disable hygiene for type expansion
macro_rules
  | `([mlir_op| mulf $op1:mlir_op_operand , $op2:mlir_op_operand : $ty:mlir_type]) => 
        `( [mlir_op| "std.mulf" ($op1, $op2) : ($ty, $ty) -> ($ty) ] )


macro_rules
  | `([mlir_op| br $op1:mlir_op_successor_arg]) => 
        `([mlir_op| "br" () [$op1] : () -> ()])

macro_rules
  | `([mlir_op| cond_br $flag: mlir_op_operand ,
          $truebb:mlir_op_successor_arg , 
          $falsebb:mlir_op_successor_arg]) => 
        `([mlir_op| "cond_br" ($flag) [$truebb, $falsebb] : ()])

-- syntax "br" 

def add0Raw := [mlir_op| "std.addi" (%op1, %op2) : (i32)]
#print add0Raw

def add0 : Op := [mlir_op| addi %c0, %c1]
#print add0


def br0 : Op := [mlir_op| br ^entry]
#print br0

def condbr0 : Op := [mlir_op| cond_br %flag, ^loopheader, ^loopexit]
#print condbr0


syntax "scf.while" "(" mlir_op_operand ")" ":" mlir_type mlir_region : mlir_op

macro_rules
  | `([mlir_op| scf.while ( $flag ) : $retty  $body]) => 
        `([mlir_op| "scf.while" ($flag) ($body) : $retty ])

def scfWhile0 := [mlir_op| "scf.while" (%x) ({ 
    ^entry: 
      -- addi %c0 %x
     "std.addi" (%op1, %op2) : (i32) 
}) : ()
]
#print scfWhile0

-- syntax "scf.if" "(" mlir_op_operand ")" ":" mlir_type mlir_region : mlir_op

-- macro_rules
--   | `([mlir_op| scf.if ( $flag ) : $retty  $body]) => 
--         `([mlir_op| "scf.if" ($flag) ($body) : $retty])

-- def scfIf0 := [mlir_op| scf.if (%x) : (i32) -> (i32) { 
--     ^entry: 
--       %z = addi %c0 %x
--       scf.while (%x) : (i32) -> (i32) { 
--         ^entry: 
--           addi %c0 %z
--       }

-- }]
-- #print scfIf0

syntax "load" mlir_op_operand "[" sepBy(mlir_op_operand, ",")  "]" : mlir_op

macro_rules
  | `([mlir_op| load $op [ $args,* ] ]) => do
        let initList <- `([[mlir_op_operand| $op]])
        let argsList <- args.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_op_operand| $x]])
        `(Op.mk "load"  $argsList [] [] (AttrDict.mk []) [mlir_type| ()])


def load0 := [mlir_op| load %foo[%ix1, %ix2] ]
#print load0
def load1 := [mlir_op| "scf.while" (%x) ({ 
    ^entry: 
     load %foo[%ix1, %ix2]
}) : ()
]

syntax "store" mlir_op_operand "[" sepBy(mlir_op_operand, ",") "]" "," mlir_op_operand : mlir_op

macro_rules
  | `([mlir_op| store $op [ $args,* ], $val ]) => do
        let initList <- `([[mlir_op_operand| $op]])
        let argsList <- args.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_op_operand| $x]])
        let stxVal <- `([mlir_op_operand| $val])
        let argsList <- `($argsList ++ [$stxVal]) -- this is terrible, I should just.. build the list! instead of building the syntax to build the list
        `(Op.mk "load"  $argsList [] [] (AttrDict.mk []) [mlir_type| ()])

def store0 := [mlir_op| store %foo[%ix1, %ix2], %val ]
#print store0
def store1 := [mlir_op| "scf.while" (%x) ({ 
    ^entry: 
     store %foo[%ix1, %ix2], %val
}) : ()
]



syntax "scf.for" "(" mlir_op_operand ")" ":" mlir_type mlir_region : mlir_op

set_option hygiene false in -- need to disable hygiene for i<blah> expansion. Otherwise it becomes i<blah>.hyg_baz
macro_rules
  | `([mlir_op| scf.for ( $flag ) : $retty  $body]) => 
        `([mlir_op| "scf.for" ($flag) ($body) : $retty])

