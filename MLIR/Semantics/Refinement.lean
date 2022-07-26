/-
## Refinement

This file defines the definition of a refinement.
-/

import MLIR.Semantics.Semantics
import MLIR.Dialects.ArithSemantics
import MLIR.Semantics.UB
open MLIR.AST

def SSAEnv.refinement (env1 env2: SSAEnv δ) : Prop :=
  ∀ v val, env2.getT v = some val -> env1.getT v = some val

-- One SSA environment is a refinement of the other if all variables
-- defined in the environment are also defined by the other first one.
-- If the SSA environment is none (corresponding to an UB), then we have
-- a refinement for all other possible SSA environment, or UB 
def refinement (env1 env2: Option R × SSAEnv δ) : Prop :=
  match env1, env2 with
  | (none, _), _ => True
  | (some r1, env1), (some r2, env2) =>
    r1 = r2 ∧ SSAEnv.refinement env1 env2
  | _, _ => False

theorem SSAEnv.refinement_set :
    refinement env1 env2 ->
    refinement (SSAEnv.set name τ val env1) (SSAEnv.set name τ val env2) := by
  simp [refinement, SSAEnv.refinement]
  intros Href name' val' Hget
  byCases H: name = name'
  . rw [SSAEnv.getT_set_eq] <;> try assumption
    rw [SSAEnv.getT_set_eq] at Hget <;> assumption
  . rw [SSAEnv.getT_set_ne] <;> try assumption
    rw [SSAEnv.getT_set_ne] at Hget <;> try assumption
    apply Href
    assumption
