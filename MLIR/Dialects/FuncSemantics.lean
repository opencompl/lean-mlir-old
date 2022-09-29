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

def funcSemanticsOp: IOp Δ →
    (Fitree (RegionE Δ +' UBE) (BlockResult Δ))
  | IOp.mk "func.return" _ args  0 _ => do
       return .Ret args
  | _ => Fitree.trigger $ UBE.Unhandled

instance: Semantics func_ where
  semantics_op := funcSemanticsOp
