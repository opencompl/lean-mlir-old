/-
## `func` dialect (stub)

We provide a definition for the `func.return` operation so we can use it during
testing without triggering undefined behavior.
-/

import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.AST
open MLIR.AST

instance func_: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

def funcSemanticsOp: IOp Δ → OpM Δ (TypedArgs Δ)
  | IOp.mk "func.return" _ args  [] _ => do
       return args
  | IOp.mk name .. => OpM.Unhandled name

instance: Semantics func_ where
  semantics_op := funcSemanticsOp
