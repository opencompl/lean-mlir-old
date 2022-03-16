import Lean.Meta.Tactic.Rewrite
import Lean.Meta.Tactic.Replace
import Lean.Elab.Tactic.Basic
import Lean.Elab.Tactic.ElabTerm
import Lean.Elab.Tactic.Location
import Lean.Elab.Tactic.Config
open Lean Meta Elab Tactic
open Lean.Elab.Term

/- Shows how to get arguments to tactics
elab "myTactic" argumentStx:term : tactic =>  do
  let goals <- getGoals
  let target <- getMainTarget
  match target.eq? with 
  | none =>  throwError "target {target} is not an equality"
  | some (equalityType, equalityLhs, equalityRhs) => 
    let maingoal <- getMainGoal
    let argumentAsTy <- Lean.Elab.Term.elabType argumentStx 

    liftMetaTactic fun mvarId => do
      -- let (h, mvarId) <- intro1P mvarId
      -- let goals <- apply mvarId (mkApp (mkConst ``Or.elim) (mkFVar h))
      let lctx <- getLCtx
      let mctx <- getMCtx
      let hypsOfType <- lctx.foldlM (init := []) (fun accum decl =>  do 
          if decl.type == equalityType 
          then return (decl.userName, decl.type) :: accum
          else return accum)
      let out := "\n====\n"
      let out := out ++ m!"-argumentStx: {argumentStx}\n"
      let out := out ++ m!"-argumentAsTy: {argumentAsTy}\n"
      let out := out ++ m!"-equalityType: {equalityType}\n"
      let out := out ++ m!"-equalityLhs: {equalityLhs}\n"
      let out := out ++ m!"-equalityRhs: {equalityRhs}\n"
      let out := out ++ m!"-hypsOfEqualityType: {hypsOfType}\n"
      -- let out := out ++ m!"-argumentStx: {argumentStx}\n"
      -- let out := out ++ m!"-mainGoal: {maingoal}\n"
      -- let out := out ++ m!"-goals: {goals}\n"
      -- let out := out ++ m!"-target: {target}\n"
      let out := out ++ "\n====\n"
      throwTacticEx `myTactic mvarId out
      return goals
-/

elab "myTactic" : tactic =>  do
  let goals <- getGoals
  let target <- getMainTarget
  match target.eq? with 
  | none =>  throwError "target {target} is not an equality"
  | some (equalityType, equalityLhs, equalityRhs) => 
    let maingoal <- getMainGoal
    liftMetaTactic fun mvarId => do
      -- let (h, mvarId) <- intro1P mvarId
      -- let goals <- apply mvarId (mkApp (mkConst ``Or.elim) (mkFVar h))
      let lctx <- getLCtx
      let mctx <- getMCtx
      let hypsOfType <- lctx.foldlM (init := []) (fun accum decl =>  do 
          if decl.type == equalityType 
          then return (decl.userName, decl.type) :: accum
          else return accum)
      let out := "\n====\n"
      let out := out ++ m!"-equalityType: {equalityType}\n"
      let out := out ++ m!"-equalityLhs: {equalityLhs}\n"
      let out := out ++ m!"-equalityRhs: {equalityRhs}\n"
      let out := out ++ m!"-hypsOfEqualityType: {hypsOfType}\n"
      -- let out := out ++ m!"-argumentStx: {argumentStx}\n"
      -- let out := out ++ m!"-mainGoal: {maingoal}\n"
      -- let out := out ++ m!"-goals: {goals}\n"
      -- let out := out ++ m!"-target: {target}\n"
      let out := out ++ "\n====\n"
      throwTacticEx `myTactic mvarId out
      return goals

-- theorem test {p: Prop} : (p ∨ p) -> p := by
--   intro h
--   apply Or.elim h
--   trace_state

-- TODO: Figure out how to extract hypotheses from goal.
theorem testSuccess : ∀ (anat: Nat) (bint: Int) (cnat: Nat) (dint: Int) (eint: Int), bint = dint := by
 intros a b c d e
 myTactic
 sorry

theorem testGoalNotEqualityMustFail  : ∀ (a: Nat) (b: Int) (c: Nat) , Nat := by
 intros a b c
 myTactic 

