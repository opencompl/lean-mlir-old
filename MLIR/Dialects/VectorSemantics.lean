/-
Define the semantics of core vector dialect operations.
-/

import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL

open MLIR.AST

instance vectorsem: Dialect Void Void (fun x => Unit) where
  name := "vector"
  iα := inferInstance
  iε := inferInstance

-- broadcast: interesting operation.
-- broadcast = replicate
-- contract: interesting operation.
-- extract
-- extractelement

def vector_semantics_op {Δ: Dialect α σ ε}: IOp Δ → OpM Δ (TypedArgs Δ)
| IOp.mk name .. => OpM.Unhandled s!"unhandled {name}"
