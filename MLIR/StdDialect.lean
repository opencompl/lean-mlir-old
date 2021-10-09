import MLIR.AST
import MLIR.EDSL

open MLIR.AST
open MLIR.EDSL

-- https://mlir.llvm.org/docs/Dialects/Standard/        

syntax "addi" mlir_op_operand mlir_op_operand : term

macro_rules
  | `(addi $op1:mlir_op_operand $op2:mlir_op_operand ) => `(mlir_op_call% "std.addi"($op1, $op2) : (i 32, i 32) )


-- | TODO: add block arguments. 
-- syntax "br" 

def add0 : Op := addi %c0 %c1
#print add0
