/-
## `arith` dialect

This file formalises part of the `arith` dialect. The goal is to showcase
operations on multiple types (with overloading) and basic reasoning. `arith`
does not have new datatypes, but it supports operations on tensors and vectors,
which are some of the most complex builtin types.

TODO: This file uses shorter operation names (without "arith.") to work around
      a normalization performance issue that is affected by the string length
See https://leanprover.zulipchat.com/#narrow/stream/270676-lean4/topic/unfold.20essentially.20loops
-/

import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Dialects.BuiltinModel
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
import Lean.Meta.Tactic.Rewrite
import Lean.Meta.Tactic.Replace
import Lean.Elab.Tactic.Basic
import Lean.Elab.Tactic.Rewrite
import Lean.Elab.Tactic.ElabTerm
import Lean.Elab.Tactic.Location
import Lean.Elab.Tactic.Config
open MLIR.AST

/-
### Dialect extensions

`arith` has no extended types or attributes.
-/

instance arith: Dialect Void Void (fun _ => Unit) where
  iα := inferInstance
  iε := inferInstance

/-
### Dialect operations

In order to support type overloads while keeping reasonably-strong typing on
operands and disallowing incorrect types in the operation arguments, we define
scalar, tensor, and vector overloads of each operation.
-/

inductive ComparisonPred :=
  | eq  | ne
  | slt | sle | sgt | sge
  | ult | ule | ugt | uge

def ComparisonPred.ofInt: Int → Option ComparisonPred
  | 0 => some eq
  | 1 => some ne
  | 2 => some slt
  | 3 => some sle
  | 4 => some sgt
  | 5 => some sge
  | 6 => some ult
  | 7 => some ule
  | 8 => some ugt
  | 9 => some uge
  | _ => none

inductive ArithE: Type → Type :=
  | CmpI: (sz: Nat) → (pred: ComparisonPred) → (lhs rhs: FinInt sz) →
          ArithE (FinInt 1)
  | AddI: (sz: Nat) → (lhs rhs: FinInt sz) →
          ArithE (FinInt sz)
  | AddT: (sz: Nat) → (D: DimList) → (lhs rhs: RankedTensor D (.int sgn sz)) →
          ArithE (RankedTensor D (.int sgn sz))
  | AddV: (sz: Nat) → (sc fx: List Nat) →
          (lhs rhs: Vector sc fx (.int sgn sz)) →
          ArithE (Vector sc fx (.int sgn sz))
  | SubI: (sz: Nat) → (lhs rhs: FinInt sz) →
          ArithE (FinInt sz)
  | NegI: (sz: Nat) → (op: FinInt sz) →
          ArithE (FinInt sz)
  | AndI: (sz: Nat) → (lhs rhs: FinInt sz) →
          ArithE (FinInt sz)
  | OrI: (sz: Nat) → (lhs rhs: FinInt sz) →
          ArithE (FinInt sz)
  | XorI: (sz: Nat) → (lhs rhs: FinInt sz) →
          ArithE (FinInt sz)

def unary_semantics_op (op: IOp Δ)
      (ctor: (sz: Nat) → FinInt sz → ArithE (FinInt sz)):
    Option (Fitree (RegionE Δ +' UBE +' ArithE) (BlockResult Δ)) :=
  match op with
  | IOp.mk name [⟨.int sgn sz, arg⟩] [] 0 _ _ => some do
      let r ← Fitree.trigger (ctor sz arg)
      return BlockResult.Next ⟨.int sgn sz, r⟩
  | IOp.mk _ _ _ _ _ _ => none

def binary_semantics_op (op: IOp Δ)
      (ctor: (sz: Nat) → FinInt sz → FinInt sz → ArithE (FinInt sz)):
    Option (Fitree (RegionE Δ +' UBE +' ArithE) (BlockResult Δ)) :=
  match op with
  | IOp.mk name [⟨.int sgn sz, lhs⟩, ⟨.int sgn' sz', rhs⟩] [] 0 _ _ => some do
      if EQ: sgn = sgn' /\ sz = sz' then
        let r ← Fitree.trigger (ctor sz lhs (EQ.2 ▸ rhs))
        return BlockResult.Next ⟨.int sgn sz, r⟩
      else
        Fitree.trigger <| UBE.DebugUB s!"{name}: incompatible operand types"
        return BlockResult.Ret []
  | IOp.mk _ _ _ _ _ _ => none

def arith_semantics_op (o: IOp Δ):
    Option (Fitree (RegionE Δ +' UBE +' ArithE) (BlockResult Δ)) :=
  match o with
  | IOp.mk "constant" [] [] 0 attrs (.fn (.tuple []) τ₁) => some <|
      match AttrDict.find attrs "value" with
      | some (.int value τ₂) =>
          if τ₁ = τ₂ then
            match τ₂ with
            | .int sgn sz => do
                -- TODO: Check range of constants
                let v := FinInt.ofInt sgn sz value
                return BlockResult.Next ⟨.int sgn sz, v⟩
            | _ => do
                Fitree.trigger $ UBE.DebugUB "non maching width of arith.const"
                return BlockResult.Ret []
          else do
                Fitree.trigger $ UBE.DebugUB "non maching type of arith.const"
                return BlockResult.Ret []
      | some _
      | none => do
            Fitree.trigger $ UBE.DebugUB "non maching type of arith.const"
            return BlockResult.Ret []

  | IOp.mk "cmpi" [ ⟨(.int sgn sz), lhs⟩, ⟨(.int sgn' sz'), rhs⟩ ] [] 0
    attrs _ => some <|
      if EQ: sgn = sgn' /\ sz = sz' then
            match attrs.find "predicate" with
            | some (.int n (.int .Signless 64)) => do
                match (ComparisonPred.ofInt n) with
                | some pred => do
                  let r ← Fitree.trigger (ArithE.CmpI sz pred lhs (EQ.2 ▸ rhs))
                  return BlockResult.Next ⟨.i1, r⟩
                | none =>
                  Fitree.trigger $ UBE.DebugUB "unable to create ComparisonPred"
                  return BlockResult.Ret []
            | some _
            | none => do
                Fitree.trigger $ UBE.DebugUB "unable to find predicate"
                return BlockResult.Ret []
      else do
        Fitree.trigger $ UBE.DebugUB "lhs, rhs, unequal sizes (cmp)"
        return BlockResult.Ret []

  | IOp.mk "negi" _ _ _ _ _ =>
      unary_semantics_op o ArithE.NegI
  | IOp.mk name _ _ _ _ _ =>
      if name = "addi" then
        binary_semantics_op o ArithE.AddI
      else if name = "subi" then
        binary_semantics_op o ArithE.SubI
      else if name = "andi" then
        binary_semantics_op o ArithE.AndI
      else if name = "ori" then
        binary_semantics_op o ArithE.OrI
      else if name = "xori" then
        binary_semantics_op o ArithE.XorI
      else
        none

def ArithE.handle {E}: ArithE ~> Fitree E := fun _ e =>
  match e with
  | AddI _ lhs rhs =>
      return lhs + rhs
  | AddT sz D lhs rhs =>
      -- TODO: Implementation of ArithE.AddT (tensor addition)
      return default
  | AddV sz sc fx lhs rhs =>
      -- TODO: Implementation of ArithE.AddV (vector addition)
      return default
  | CmpI _ pred lhs rhs =>
      let b: Bool :=
        match pred with
        | .eq  => lhs = rhs
        | .ne  => lhs != rhs
        | .slt => lhs.toSint <  rhs.toSint
        | .sle => lhs.toSint <= rhs.toSint
        | .sgt => lhs.toSint >  rhs.toSint
        | .sge => lhs.toSint >= rhs.toSint
        | .ult => lhs.toUint <  rhs.toUint
        | .ule => lhs.toUint <= rhs.toUint
        | .ugt => lhs.toUint >  rhs.toUint
        | .uge => lhs.toUint >= rhs.toUint
      return FinInt.ofInt .Signless 1 (if b then 1 else 0)
  | SubI _ lhs rhs =>
      return lhs - rhs
  | NegI _ op =>
      return -op
  | AndI _ lhs rhs =>
      return lhs &&& rhs
  | OrI _ lhs rhs =>
      return lhs ||| rhs
  | XorI _ lhs rhs =>
      return lhs ^^^ rhs

instance: Semantics arith where
  E := ArithE
  semantics_op := arith_semantics_op
  handle := ArithE.handle

/-
### Basic examples
-/

private def cst1: BasicBlock arith := [mlir_bb|
  ^bb:
    %true = "constant" () {value = 1: i1}: () -> i1
    %false = "constant" () {value = 0: i1}: () -> i1
    %r1 = "constant" () {value = 25: i32}: () -> i32
    %r2 = "constant" () {value = 17: i32}: () -> i32
    %r = "addi" (%r1, %r2): (i32, i32) -> i32
    %s = "subi" (%r2, %r): (i32, i32) -> i32
    %b1 = "cmpi" (%r, %r1) {predicate = 5 /- sge -/}: (i32, i32) -> i1
    %b2 = "cmpi" (%r2, %r) {predicate = 8 /- ugt -/}: (i32, i32) -> i1
]

-- #eval run (Δ := arith) ⟦cst1⟧ (SSAEnv.empty (δ := arith))


/-
### Rewriting heorems
-/

open FinInt(mod2)

/-===  n+m  <-->  m+n  ===-/

private def th1_org: BasicBlockStmt arith := [mlir_bb_stmt|
  %r = "addi"(%n, %m): (i32, i32) -> i32
]
private def th1_out: BasicBlockStmt arith := [mlir_bb_stmt|
  %r = "addi"(%m, %n): (i32, i32) -> i32
]

/- private theorem th1:
  forall (n m: FinInt 32),
    run ⟦th1_org⟧ (SSAEnv.One [ ("n", ⟨.i32, n⟩), ("m", ⟨.i32, m⟩) ]) =
    run ⟦th1_out⟧ (SSAEnv.One [ ("n", ⟨.i32, n⟩), ("m", ⟨.i32, m⟩) ]) := by
  intros n m
  simp [Denote.denote]
  simp [run, th1_org, th1_out, denoteBBStmt, denoteOp]
  simp [interp_ub, SSAEnv.get]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [Semantics.handle, ArithE.handle, SSAEnv.get]; simp_itree
  simp [FinInt.add_comm] -/

/- LLVM InstCombine: `C-(X+C2) --> (C-C2)-X`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L1794 -/


private def th2_org: BasicBlock arith := [mlir_bb|
  ^bb:
    %t = "addi"(%X, %C2): (i32, i32) -> i32
    %r = "subi"(%C, %t): (i32, i32) -> i32
]
private def th2_out: BasicBlock arith := [mlir_bb|
  ^bb:
    %t = "subi"(%C, %C2): (i32, i32) -> i32
    %r = "subi"(%t, %X): (i32, i32) -> i32
]
private def th2_input (C X C2: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("C", ⟨.i32, C⟩), ("X", ⟨.i32, X⟩), ("C2", ⟨.i32, C2⟩)
]



/- LLVM InstCombine: `~X + C --> (C-1) - X`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L882 -/


private def th3_org: BasicBlock arith := [mlir_bb|
  ^bb:
    %o = "constant"() {value = 1: i32}: () -> i32
    %m = "negi"(%o): (i32) -> i32
    %r = "addi"(%m, %C): (i32, i32) -> i32
]

private def th3_out: BasicBlock arith := [mlir_bb|
  ^bb:
    %o = "constant"() {value = 1: i32}: () -> i32
    %t = "subi"(%C, %o): (i32, i32) -> i32
    %r = "subi"(%t, %X): (i32, i32) -> i32
]
private def th3_input (C X: FinInt 32): SSAEnv arith := SSAEnv.One [
    ("C", ⟨.i32, C⟩), ("X", ⟨.i32, X⟩)
]

theorem Fitree.bind_Ret: Fitree.bind (Fitree.Ret r) k = k r := rfl

theorem Fitree.bind_ret: Fitree.bind (Fitree.ret r) k = k r := rfl

theorem Fitree.bind_Vis: Fitree.bind (Fitree.Vis e k) k' =
  Fitree.Vis e (fun r => bind (k r) k') := rfl

/-
private theorem th3_left_timeout: forall (C X: FinInt 32),
    run (denoteBB _ th3_org) (th3_input C X) = x := by

  dsimp [th3_input, th3_org, th3_out, run]
  unfold denoteBB, denoteBBStmt, denoteOp
  simp [List.zip, List.mapM, bind]
  dsimp [Semantics.semantics_op, arith_semantics_op]
  dsimp_itree
  dsimp [AttrDict.find, List.find?, AttrEntry.key, AttrEntry.value]
  simp
  dsimp [interp, pure]
  simp [Fitree.bind_ret]
  dsimp [interp_ub]
  simp_itree
--  rw [Fitree.bind_Vis]
-/

/-
elab "cbn" : tactic => do 
  let t <- Core.transform
  let target <- getMainTarget
  let target_unfolded? <- Meta.delta? target
  match target_unfolded? with 
  | some unfolded => 
     let new <- whnf unfolded
     -- replaceMainGoal
  | none => throwError "nothing to δ reduce at {target}"
-/

-- Stolen from conv/basic.lean

open Lean Elab Meta Tactic in 
def getLhsRhsCore (mvarId : MVarId) : MetaM (Expr × Expr) :=
  withMVarContext mvarId do
    let some (_, lhs, rhs) ← matchEq? (← getMVarType mvarId) | throwError "invalid 'conv' goal"
    return (lhs, rhs)


open Lean Elab Meta Tactic in 
def getLhsRhs : TacticM (Expr × Expr) := do
  getLhsRhsCore (← getMainGoal)

open Lean Elab Meta Tactic in 
def getRhs : TacticM Expr :=
  return (← getLhsRhs).2


open Lean Elab Meta Tactic in 
def getLhs : TacticM Expr :=
  return (← getLhsRhs).1

open Lean Elab Meta Tactic in 
def updateLhs (lhs' : Expr) (h : Expr) : TacticM Unit := do
  let rhs ← getRhs
  let newGoal ← mkFreshExprSyntheticOpaqueMVar (mkLHSGoal (← mkEq lhs' rhs))
  assignExprMVar (← getMainGoal) (← mkEqTrans h newGoal)
  replaceMainGoal [newGoal.mvarId!]

open Lean Elab Meta Tactic in 
def changeLhs (lhs' : Expr) : TacticM Unit := do
  let rhs ← getRhs
  liftMetaTactic1 fun mvarId => do
    replaceTargetDefEq mvarId ((← mkEq lhs' rhs))



open Lean Elab Meta Tactic in 
def unfoldIfUseful (e: Expr) (names: Array Name): TacticM TransformStep := do
  -- let e? <- Meta.delta? e (fun name => names.contains name)
  let e? <- names.foldlM (init := Option.none) (fun accum name => do
      match accum with 
      | .some e => return .some e 
      | .none => match (<- Meta.delta? e (. == name)) with 
                 | .some e => dbg_trace "ran {name}"; return .some e
                 | .none => return .none) 
  match e? with 
  | .some e => do 
    -- let e' <- whnf e
    return TransformStep.done e
  | .none => return TransformStep.visit e

open Lean Elab Meta Tactic in
elab "cbn!" "[" rewrites:ident,* "]"  : tactic => withMainContext do
  let target <- getLhs
  let declNames <- rewrites.getElems.mapM  resolveGlobalConstNoOverload
  let new <- (Core.transform target (pre := unfoldIfUseful (names := declNames)))
  let new <- zetaReduce new
  changeLhs new



/-
private theorem th2:
  forall (C X C2: FinInt 32),
    (run (denoteBB _ th2_org) (th2_input C X C2) |>.snd.get "r" .i32) =
    (run (denoteBB _ th2_out) (th2_input C X C2) |>.snd.get "r" .i32) := by
  intros C X C2
  simp [th2_input, th2_org, th2_out]
  simp [run, denoteBB, denoteBBStmt, denoteOp]; simp_itree
  simp [interp_ub]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [Semantics.handle, ArithE.handle, SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  sorry
-/

open Lean Elab Meta Tactic in 
elab "cbn_itree" "[" rewrites:term,* "]" : tactic => withMainContext do
  -- TODO: Also handle .lemmaNames, not just unfolding!
  let rewriteNames <- rewrites.getElems.mapM  resolveGlobalConstNoOverload
  let unfoldLemmas <- (← SimpItreeExtension.getTheorems).toUnfold.foldM
  --  (init := #[]) (fun acc n => do acc.push (<- resolveGlobalConstNoOverload n))
     (init := #[]) (fun acc n => return acc.push n)
  let treeLemmas := #[`Member.inject,
    `StateT.bind, `StateT.pure, `StateT.lift,
    `OptionT.bind, `OptionT.pure, `OptionT.mk, `OptionT.lift,
    `bind, `pure, `cast_eq, `Eq.mpr]
  let target <- getLhs
  dbg_trace unfoldLemmas
  let smallUnfoldLemmas : Array Name :=
            #[`UBE.handleSafe,
              `Fitree.run,
              `interp,
              `SSAEnv.get?,
              `interp',
              `MLIR.AST.MLIRType.eval,
              `UBE.handle,
              `Fitree.ret,
              `Fitree.translate,
              `writerT_defaultHandler,
              `Fitree.trigger,
              `optionT_defaultHandler, 
              `Fitree.bind,
              `interp_state,
              `Fitree.case_, `SSAEnvE.handle, `interp_writer, `SSAEnv.set?, `UBE.handle!,
              `stateT_defaultHandler,
              `interp_option]
  let new <- Meta.transform target (pre := unfoldIfUseful
                                            (names := unfoldLemmas ++ treeLemmas ++ rewriteNames))
  let new <- zetaReduce new
  changeLhs new

#check Fitree.run
private theorem th2_cbn:
  forall (C X C2: FinInt 32),
    (run (denoteBB _ th2_org) (th2_input C X C2) |>.snd.get "r" .i32) =
    (run (denoteBB _ th2_out) (th2_input C X C2) |>.snd.get "r" .i32) := by
  intros C X C2
  cbn! [th2_input, th2_org, th2_out]
  cbn! [run, denoteBB, denoteBBStmt, denoteOp];
  unfold Fitree.run 
  -- cbn_itree []
  -- cbn_itree []
  -- cbn_itree [SSAEnv.get]
  -- cbn_itree [SSAEnv.get]
  
  
  -- cbn_itree [Fitree.run]
  
   

 /-
  cbn_itree;
  simp [interp_ub]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [Semantics.handle, ArithE.handle, SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
-/
  sorry



/-
 private theorem th3:
  forall (C X: FinInt 32),
    (run (denoteBB _ th3_org) (th3_input C X) |>.snd.get "r" .i32) =
    (run (denoteBB _ th3_out) (th3_input C X) |>.snd.get "r" .i32) := by

/-
  intros C X
  dsimp [th3_input, th3_org, th3_out, run]
  unfold denoteBB, denoteBBStmt, denoteOp; simp
  save
  -- Fully simplify the semantics
  dsimp [Semantics.semantics_op, arith_semantics_op, List.zip, List.mapM]
  dsimp_itree
  dsimp [AttrDict.find, List.find?, AttrEntry.key, AttrEntry.value]
  simp
  dsimp [interp, pure] -/
--  simp [Fitree.bind]
--  dsimp_itree
/-  simp_itree
  simp [AttrDict.find, List.find?, AttrEntry.key, AttrEntry.value]
  simp_itree
  save
  -- Interpret events
  dsimp [interp_ub]
  dsimp_itree
  save
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]
  simp_itree
  save -/
/-
  simp [SSAEnv.get]; simp_itree
  simp [Semantics.handle, ArithE.handle, SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree -/

-/

/-
theorem FinInt.sub_add_dist: forall (C X C2: FinInt sz),
    C - (X + C2) = (C - C2) - X := by
  intros C X C2
  apply eq_of_toUint_cong2
  simp [cong2, FinInt.sub_toUint, FinInt.add_toUint]

  apply mod2_equal
  simp [Int.sub_add_dist]
  sorry_arith -- rearrange terms

theorem FinInt.comp_add: sz > 0 → forall (X C: FinInt sz),
    comp X + C = (C - FinInt.ofUint sz 1) - X := by
  intros h_sz X C
  apply eq_of_toUint_cong2
  simp [cong2, FinInt.add_toUint, FinInt.comp_toUint, FinInt.sub_toUint]
  simp [FinInt.toUint_ofUint]
  have h: mod2 1 sz = 1 := mod2_idem ⟨by decide, by sorry_arith⟩
  simp [h]
  sorry_arith -- eliminate 2^sz in lhs, then mod2_equal

private theorem mod2_equal: x = y → mod2 x n = mod2 y n :=
  fun | .refl _ => rfl

-/
