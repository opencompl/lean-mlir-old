
-- TODO: I would hope that some tactic is already available for this?
-- TODO: I don't know how to execute some tactics only in first goal, and some in second goal
-- TODO: How can we have arbitrary terms instead of idents? term do not work

/- `byCases H: x = y` splits the goal into one goal that has `x = y`, and one that hase `x ≠ y`.
 `H` will contain the new hypothesis, and in the equality case, `x` will be substituted by `y`,
 or `y` by `x` depending on which one will work.
 `simp [H]` will be called on both subgoals. -/
macro "byCases" Hname:ident ":" lhs:ident "=" rhs:ident : tactic =>   
  `(tactic| apply Decidable.byCases (p:= $lhs = $rhs) <;> (intros $Hname) <;> (simp [$Hname:ident]) <;> (try subst $lhs) <;> (try subst $rhs))
