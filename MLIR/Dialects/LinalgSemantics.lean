/-
Define the semantics of core Linalg operations.
-/
import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
open MLIR.AST

instance linalg: Dialect Void Void (fun x => Unit) where
  name := "linalg"
  iα := inferInstance
  iε := inferInstance

def linalg_region_adaptor2d:

def linalg_semantics_op: IOp Δ → OpM Δ (TypedArgs Δ)
  | IOp.mk "linalg.extractslice2d" _ [⟨.tensor, t⟩]  [r] dict => sorry
  | IOp.mk "linalg.insertslice2d" _ [⟨.tensor, t⟩]  [r] dict => sorry
  | IOp.mk "linalg.fill2d" _ [⟨.tensor, t⟩]  [r] dict => sorry
  | IOp.mk "linalg.tile1d" _ [⟨.tensor, t⟩]  [r] dict => sorry
  | IOp.mk "linalg.transpose2d"   _ [⟨.tensor, t⟩]  [r] dict => sorry
  | IOp.mk "linalg.generic2d" _ [⟨.tensor, t⟩]  [r] dict => sorry
  | IOp.mk "linalg.parallel2d" _ [⟨.tensor, t⟩]  [r] dict => do
      return []
  | _ => OpM.Unhandled "unhandled linalg.generic"
