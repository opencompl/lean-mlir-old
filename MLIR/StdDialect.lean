import MLIR.AST
import MLIR.EDSL

open MLIR.AST
open MLIR.EDSL

-- def addi (x: SSAVal) (y: SSAVal): SSAVal := (mlir_op_call% "std.addi" (x, y) : (i 32, i 32) -> (i 32) )
