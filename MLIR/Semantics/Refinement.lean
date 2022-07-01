/-
## Refinement

This file defines the definition of a refinement.
For now, this is only defined for the arith dialect, but this 
can hopefully be defined for arbitrary semantics.
-/

import MLIR.Semantics.Semantics
import MLIR.Dialects.ArithSemantics
import MLIR.Semantics.UB
open MLIR.AST

def SSAEnv.refinement (env1 env2: SSAEnv arith) : Prop :=
  ∀ v val, env2.getT v = some val -> env1.getT v = some val

-- One SSA environment is a refinement of the other if all variables
-- defined in the environment are also defined by the other first one.
-- If the SSA environment is none (corresponding to an UB), then we have
-- a refinement for all other possible SSA environment, or UB 
def refinement (env1 env2: Option R × SSAEnv arith) : Prop :=
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

theorem arithRefinement:
    ∀ (env1 env2: SSAEnv arith),
    SSAEnv.refinement env1 env2 ->
    ∀ (res: Option SSAVal) (op: Op arith) (t: Fitree (SSAEnvE arith +' Semantics.E arith) _),
    Semantics.semantics_op res op = some t ->
    refinement (run (Fitree.translate Member.inject t) env1) (run (Fitree.translate Member.inject t) env2) := by
  intros env1 env2 Href res op t Hsem
  simp [Semantics.semantics_op, arith_semantics_op] at Hsem
  split at Hsem
  . split at Hsem <;> try contradiction
    split at Hsem <;> try contradiction
    split at Hsem <;> try contradiction
    simp at Hsem
    rw [←Hsem]
    cases res with
    | none =>
      simp_itree; simp [run]; simp_itree; simp [refinement]
      assumption
    | some val =>
      simp_itree
      simp [run]
      simp_itree
      simp [refinement] at *
      apply SSAEnv.refinement_set <;> try assumption
  . split at Hsem <;> try contradiction
    split at Hsem <;> try contradiction
    split at Hsem <;> try contradiction
    simp at Hsem
    sorry
  . sorry
  . sorry
  -- cases op with | mk name args bbs regions attrs ty => 
  
  



section ArithPreservation
variable [Hsem: Semantics arith]

variable (HCorrectsem:
           ∀ (env1 env2: SSAEnv arith),
           SSAEnv.refinement env1 env2 ->
           ∀ (res: Option SSAVal) (op: Op arith) (t: Fitree (SSAEnvE arith +' Semantics.E arith) _),
           Semantics.semantics_op res op = some t ->
           refinement (run (Fitree.translate Member.inject t) env1) (run (Fitree.translate Member.inject t) env2))

mutual
theorem arithPreservesRefinementOp :
  ∀ (env1 env2: SSAEnv arith),
    SSAEnv.refinement env1 env2 ->
    ∀ (res: Option SSAVal) (op: Op arith),
    refinement (run (semantics_op! res op) env1) (run (semantics_op! res op) env2) := by
  intros env1 env2 Href res op
  simp [semantics_op!, Semantics.semantics_op]
  cases Ht: (Semantics.semantics_op (self := Hsem) res op)
  . simp [run, interp_ub, interp]
    simp_itree
    simp [refinement]
  . simp
    apply HCorrectsem <;> assumption

theorem arithPreservesRefinementBBStmt :
  ∀ (env1 env2: SSAEnv arith),
    SSAEnv.refinement env1 env2 ->
    ∀ (prog: BasicBlockStmt arith),
    refinement (run ⟦prog⟧ env1) (run ⟦prog⟧ env2) := by
  intros env1 env2 Href prog
  cases prog <;> apply arithPreservesRefinementOp _ _ Href
end

end ArithPreservation


section ArithPreservation'
mutual
theorem arithPreservesRefinementOp' :
  ∀ (env1 env2: SSAEnv arith),
    SSAEnv.refinement env1 env2 ->
    ∀ (res: Option SSAVal) (op: Op arith),
    refinement (run (S:=arithSemantics) (semantics_op! (S:=arithSemantics) res op) env1) (run (S:=arithSemantics) (semantics_op! (S:=arithSemantics) res op) env2) := by
  intros env1 env2 Href res op
  simp [semantics_op!, Semantics.semantics_op]

theorem arithPreservesRefinementBBStmt' :
  ∀ (env1 env2: SSAEnv arith),
    SSAEnv.refinement env1 env2 ->
    ∀ (prog: BasicBlockStmt arith),
    refinement (run (S:=arithSemantics) (Denote.denote (S:=arithSemantics) prog) env1) (run (S:=arithSemantics) (Denote.denote (S:=arithSemantics) prog) env2) := by
  intros env1 env2 Href prog
  cases prog 
  . simp [Denote.denote, semantics_bbstmt]
    sorry
  . sorry
  --cases prog <;> apply arithPreservesRefinementOp' _ _ Href
end

end ArithPreservation'