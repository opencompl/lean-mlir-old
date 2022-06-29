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

def funcSemanticsOp:
    IOp func_ → Fitree (RegionE +' UBE +' SSAEnvE func_ +' PVoid) (BlockResult func_)

  | IOp.mk "func.return" args [] 0 _ _ =>
       return .Ret args
  | _ => do
    Fitree.trigger $ UBE.DebugUB "unknown func op"
    return BlockResult.Ret []

instance: Semantics func_ where
  E := PVoid
  semantics_op := funcSemanticsOp
  handle := fun _ e => nomatch e
