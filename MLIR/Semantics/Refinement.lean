/-
## Refinement
This file defines a refinement of semantics.
-/

import MLIR.Semantics.Semantics
import MLIR.Semantics.UB
open MLIR.AST

/-
### Refinement of SSA environment

One SSA environment `e1` is a refinement of an environment `e2` if all variables
defined in `e2` are also defined by `e1`, and have the same values.
-/

def SSAEnv.refines (env1 env2: SSAEnv δ) : Prop :=
  ∀ v val, env2.getT v = some val -> env1.getT v = some val

theorem SSAEnv.refines_set :
    refines env1 env2 ->
    refines (SSAEnv.set name τ val env1) (SSAEnv.set name τ val env2) := by
  simp [refines, SSAEnv.refines]
  intros Href name' val' Hget
  by_cases H: name = name'

  . subst H;
    rw [SSAEnv.getT_set_eq] <;> try assumption
    rw [SSAEnv.getT_set_eq] <;> try assumption
    intros K; assumption;
  . rw [SSAEnv.getT_set_ne] <;> try assumption
    rw [SSAEnv.getT_set_ne] <;> try assumption
    intros K;
    apply Href; assumption;


/-
### Refinement of programs

A program `p1` refines a program `p2` if `p1` succeed whenever `p2` succeed,
with the same return value, and an environment refining `p2` environment.
-/

def TopM.refines {Δ: Dialect α' σ' ε'} {R} (t1 t2: TopM Δ R) :=
  ∀ env,
  match run t2 env, run t1 env with
  | .error _, _ => true
  | .ok (v2, env2), .ok (v1, env1) => v1 = v2 ∧ env1.refines env2
  | _, _ => false
