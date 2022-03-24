
import MLIR.Doc
import MLIR.AST
import MLIR.EDSL

open  MLIR.EDSL
open MLIR.Doc
open IO

-- | write an op into the path
def o: Op := [mlir_op|
    "builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<2xi32> ):
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()


] 
-- | main program
def main : IO Unit :=
    let str := Pretty.doc o
    FS.writeFile "test-alias-analysis-modref.out.txt" str
