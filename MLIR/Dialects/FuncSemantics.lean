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
    Option (Fitree (RegionE Δ +' UBE  +' Void1) (BlockResult Δ))
  | IOp.mk "func.return" args [] 0 _ _ => some <|
       return .Ret args
  | _ => none

instance: Semantics func_ where
  E := Void1
  semantics_op := funcSemanticsOp
  handle := fun _ e => nomatch e
