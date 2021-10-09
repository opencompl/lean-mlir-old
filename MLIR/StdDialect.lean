import MLIR.AST
import MLIR.EDSL

open MLIR.AST
open MLIR.EDSL

-- https://mlir.llvm.org/docs/Dialects/Standard/        

syntax "addi" mlir_op_operand mlir_op_operand : term
syntax "br" mlir_op_successor_arg : term
syntax "cond_br" mlir_op_operand "," mlir_op_successor_arg "," mlir_op_successor_arg : term

macro_rules
  | `(addi $op1:mlir_op_operand $op2:mlir_op_operand ) => 
        `(mlir_op% "std.addi" ($op1, $op2) : (i 32, i 32) )
macro_rules
  | `(br $op1:mlir_op_successor_arg) => 
        `(mlir_op% "br" () [$op1] : () -> ())

macro_rules
  | `(cond_br $flag: mlir_op_operand ,
          $truebb:mlir_op_successor_arg , 
          $falsebb:mlir_op_successor_arg) => 
        `(mlir_op% "cond_br" ($flag) [$truebb, $falsebb] : (i 1) -> () )

-- | TODO: add block arguments. 
-- syntax "br" 

def add0 : Op := addi %c0 %c1
#print add0

def br0 : Op := br ^entry
#print br0

def condbr0 : Op := cond_br %flag, ^loopheader, ^loopexit
#print condbr0


syntax "while" "(" mlir_op_operand ")" ":" mlir_type mlir_region : term

macro_rules
  | `(while ( $flag ) : $retty  $body ) => 
        `(mlir_op% "while" ($flag) ($body) : $retty )

def while0 := while (%x) : (i 32) -> (i 32) { 
    ^entry: 
      -- addi %c0 %x
}
#print while0