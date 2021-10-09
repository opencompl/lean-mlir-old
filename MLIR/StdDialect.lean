import MLIR.AST
import MLIR.EDSL

open MLIR.AST
open MLIR.EDSL

-- https://mlir.llvm.org/docs/Dialects/Standard/        

syntax "addi" mlir_op_operand mlir_op_operand : mlir_op

syntax "br" mlir_op_successor_arg : mlir_op
syntax "cond_br" mlir_op_operand "," mlir_op_successor_arg "," mlir_op_successor_arg : mlir_op

macro_rules
  | `(mlir_op% addi $op1:mlir_op_operand $op2:mlir_op_operand ) => 
        `(mlir_op% "std.addi" ($op1, $op2) : (i 32, i 32) )
macro_rules
  | `(mlir_op% br $op1:mlir_op_successor_arg) => 
        `(mlir_op% "br" () [$op1] : () -> ())

macro_rules
  | `(mlir_op% cond_br $flag: mlir_op_operand ,
          $truebb:mlir_op_successor_arg , 
          $falsebb:mlir_op_successor_arg) => 
        `(mlir_op% "cond_br" ($flag) [$truebb, $falsebb] : (i 1) -> () )

-- | TODO: add block arguments. 
-- syntax "br" 

def add0 : Op := (mlir_op% addi %c0 %c1)
#print add0

def br0 : Op := (mlir_op% br ^entry)
#print br0

def condbr0 : Op := (mlir_op% cond_br %flag, ^loopheader, ^loopexit)
#print condbr0


syntax "scf.while" "(" mlir_op_operand ")" ":" mlir_type mlir_region : mlir_op

macro_rules
  | `(mlir_op% scf.while ( $flag ) : $retty  $body ) => 
        `(mlir_op% "scf.while" ($flag) ($body) : $retty )

def scfWhile0 := (mlir_op% scf.while (%x) : (i 32) -> (i 32) { 
    ^entry: 
      addi %c0 %x
})
#print scfWhile0

syntax "scf.if" "(" mlir_op_operand ")" ":" mlir_type mlir_region : mlir_op

macro_rules
  | `(mlir_op% scf.if ( $flag ) : $retty  $body ) => 
        `(mlir_op% "scf.if" ($flag) ($body) : $retty )

def scfIf0 := (mlir_op% scf.if (%x) : (i 32) -> (i 32) { 
    ^entry: 
      %z = addi %c0 %x
      scf.while (%x) : (i 32) -> (i 32) { 
        ^entry: 
          addi %c0 %z
      }

})
#print scfIf0