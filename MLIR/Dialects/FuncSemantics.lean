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

def func_semantics_op {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} (ret: Option SSAVal):
    Op Gδ → Option (Fitree (SSAEnvE Gδ +' PVoid) (BlockResult Gδ))

  | Op.mk "func.return" args [] [] _ (.fn (.tuple τs) (.tuple [])) =>
      if args.length != τs.length then
        none
      else some do
        return .Ret (List.zip args τs)

  | _ => none

instance: Semantics func_ where
  E := PVoid
  semantics_op := func_semantics_op
  handle := fun _ e => nomatch e
