import Lean
open Lean.Elab.Tactic

-- TODO: I would hope that some tactic is already available for this?
-- TODO: I don't know how to execute some tactics only in first goal, and some in second goal
-- TODO: How can we have arbitrary terms instead of idents? term do not work

/- `byCases H: x = y` splits the goal into one goal that has `x = y`, and one that hase `x ≠ y`.
 `H` will contain the new hypothesis, and in the equality case, `x` will be substituted by `y`,
 or `y` by `x` depending on which one will work.
 `simp [H]` will be called on both subgoals. -/
macro "byCases" Hname:ident ":" lhs:ident "=" rhs:ident : tactic =>   
  `(tactic| apply Decidable.byCases (p:= $lhs = $rhs) <;> (intros $Hname) <;> (simp [$Hname:ident]) <;> (try subst $lhs) <;> (try subst $rhs))

-- Apply a tactic, and fails if the tactic did not changed the goal or
-- the hypotheses.
elab  "progress " tac:tactic : tactic => do
    let goal ← getMainTarget
    (← getMainGoal).withContext do
     let lctx <- Lean.getLCtx
    --let goals := (← getGoals).map λ g => Lean.instantiateMVars g.type
     evalTactic tac
     (← getMainGoal).withContext do
       let lctx' <- Lean.getLCtx
       let goal' ← getMainTarget
       let lctxDeclExprs := lctx.decls.toArray.map (Option.map Lean.LocalDecl.type)
       let lctx'DeclExprs := lctx'.decls.toArray.map (Option.map Lean.LocalDecl.type)
       if goal == goal' && lctxDeclExprs == lctx'DeclExprs then
         throwError "no progress made{(← getMainGoal).name}"
