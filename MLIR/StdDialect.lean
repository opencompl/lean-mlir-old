import MLIR.AST
import MLIR.EDSL

open MLIR.AST
open MLIR.EDSL

-- https://mlir.llvm.org/docs/Dialects/Standard/        

syntax "addi" mlir_op_operand mlir_op_operand : term
syntax "br" mlir_op_successor_arg : term

macro_rules
  | `(addi $op1:mlir_op_operand $op2:mlir_op_operand ) => 
        `(mlir_op% "std.addi" ($op1, $op2) : (i 32, i 32) )
macro_rules
  | `(br $op1:mlir_op_successor_arg) => 
        `(mlir_op% "br" () [$op1] : () -> ())

-- | TODO: add block arguments. 
-- syntax "br" 

def add0 : Op := addi %c0 %c1
#print add0

def br0 : Op := br ^entry
#print br0
