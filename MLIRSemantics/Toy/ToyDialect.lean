import MLIRSemantics.Toy.Toy
import MLIRSemantics.Fitree
import MLIRSemantics.Verifier
import MLIRSemantics.SSAEnv
import MLIRSemantics.InvalidOp
import MLIRSemantics.Util.Metagen

import MLIR.AST
import MLIR.EDSL

import Lean

open MLIR.AST


/- To be automatically generated -/

set_option hygiene false in
genInductive ToyOp #[
  OpSpec.mk "Constant" `(
    (D: DimList) → (Hknown: D.known) →
    (e: TensorElem) → (τ: MLIRTy) → (Htype: e.hasType τ) →
    (Hcompat: e.rankCompatibleWith D τ) →
    ToyOp (RankedTensor τ.eval D)),
  OpSpec.mk "Transpose" `(
    (α: Type) → (n m: Nat) →
    RankedTensor α [Dimension.Known n, Dimension.Known m] →
    ToyOp (RankedTensor α [Dimension.Known m, Dimension.Known n])),
  OpSpec.mk "Reshape" `(
    (α: Type) → (D D': DimList) → (H: D.known) → (H': D'.known) →
    (Hprod: D'.prod = D.prod) →
    RankedTensor α D →
    ToyOp (RankedTensor α D'))
]

/- To be automatically generated (hopefully; basically this is the
   verification stuff) -/

def toy_semantics_op (ret_name: Option SSAVal):
      Op → Fitree (InvalidOpE +' SSAEnvE +' ToyOp) Unit

  | Op.mk "toy.constant" [] [] [] attrs
        (MLIRTy.fn (MLIRTy.tuple []) (MLIRTy.tensorRanked D₁ τ₁)) =>
      match AttrDict.find attrs "value" with
      | some (AttrVal.dense elem (MLIRTy.tensorRanked D₂ τ₂)) =>
          if H: D₁ = D₂ ∧ DimList.known D₁ ∧ τ₁ = τ₂ ∧ elem.hasType τ₁ then
            match Heq: elem, τ₁ with
            | TensorElem.int i, MLIRTy.int 32 => do
                let t ← Fitree.trigger (ToyOp.Constant D₁ H.2.1 elem
                  (MLIRTy.int 32) (by simp [Heq, H.2.2.2])
                  (TensorElem.rankCompatibleWith.UniformInt i 32 Heq));
                SSAEnv.set? (MLIRTy.tensorRanked D₁ (MLIRTy.int 32)) ret_name t
            | elem, τ₁ => do
                if Hshape: elem.hasShape (DimList.default_refinement D₁) then
                  let t ← Fitree.trigger (ToyOp.Constant D₁ H.2.1 elem τ₁
                    H.2.2.2 (TensorElem.rankCompatibleWith.HasShape
                     (DimList.default_refinement D₁) _ Hshape
                     (default_refinement_refines D₁)));
                  SSAEnv.set? (MLIRTy.tensorRanked D₁ τ₁) ret_name t
                else
                  Fitree.trigger InvalidOpE.InvalidOp
          else
            Fitree.trigger InvalidOpE.InvalidOp
      | _ =>
          Fitree.trigger InvalidOpE.InvalidOp

  | Op.mk "toy.transpose" [t_name] [] [] _ (MLIRTy.fn τ₁ τ₂) =>
      match τ₁ with
      | MLIRTy.tensorRanked [Dimension.Known n, Dimension.Known m] τ => do
          let t ← Fitree.trigger (@SSAEnvE.Get (MLIRTy.tensorRanked
                  [Dimension.Known n, Dimension.Known m] τ) _ t_name);
          let t' ← Fitree.trigger (ToyOp.Transpose τ.eval n m t);
          SSAEnv.set? (MLIRTy.tensorRanked [Dimension.Known m,
                       Dimension.Known n] τ)
            ret_name t'
      | _ =>
          Fitree.trigger InvalidOpE.InvalidOp

  | Op.mk "toy.reshape" [t_name] [] [] _ (MLIRTy.fn τ₁ τ₂) =>
      match τ₁, τ₂ with
      | MLIRTy.tensorRanked D σ₁, MLIRTy.tensorRanked D' σ₂ =>
          if H: σ₁ = σ₂
             ∧ DimList.known D
             ∧ DimList.known D'
             ∧ DimList.prod D' = DimList.prod D then do
            let t ← Fitree.trigger (@SSAEnvE.Get (MLIRTy.tensorRanked D σ₁) _
                    t_name);
            let t' ← Fitree.trigger (ToyOp.Reshape σ₁.eval D D'
                     H.2.1 H.2.2.1 H.2.2.2 t);
            let t': RankedTensor σ₂.eval D' := cast (by rw [H.1]) t';
            SSAEnv.set? (MLIRTy.tensorRanked D' σ₂) ret_name t'
          else
            Fitree.trigger InvalidOpE.InvalidOp
      | _, _ =>
          Fitree.trigger InvalidOpE.InvalidOp

  | _ =>
      Fitree.trigger InvalidOpE.InvalidOp

def toy_semantics_bbstmt:
      BasicBlockStmt → Fitree (InvalidOpE +' SSAEnvE +' ToyOp) Unit
  | BasicBlockStmt.StmtAssign val _ op =>
      toy_semantics_op (some val) op
  | BasicBlockStmt.StmtOp op =>
      toy_semantics_op none op

-- TODO: toy_semantics_bb: handle basic block arguments
@[simp]
def toy_semantics_bb:
      BasicBlock → Fitree (InvalidOpE +' SSAEnvE +' ToyOp) Unit
  | BasicBlock.mk name args [] =>
      Fitree.ret ()
  | BasicBlock.mk name args (op1::ops) =>
      List.foldr (fun t acc => Fitree.bind acc (fun _ => t))
                 (toy_semantics_bbstmt op1)
                 (ops.map toy_semantics_bbstmt)

/- Manually specified: ToyOp event handler -/

def ToyOp.handle {E}: ToyOp ~> Fitree E :=
  fun _ e => match e with
  | ToyOp.Constant D Hknown elem τ Htype Hcompat =>
      return RankedTensor.ofTensorElem D elem Htype Hcompat
  | ToyOp.Transpose α n m t =>
      return transpose t
  | ToyOp.Reshape α D D' H H' Hprod t =>
      return reshape D' H H' Hprod t

-- Interpretation in context

def interp_toy {E} (t: Fitree (ToyOp +' E) R): Fitree E R :=
  interp (case_ ToyOp.handle (fun T => @Fitree.trigger E E T _)) t

@[simp]
def run_toy (t: Fitree (InvalidOpE +' SSAEnvE +' ToyOp) Unit) (env: SSAEnv):
    Fitree PVoid (Unit × SSAEnv) :=
  interp ToyOp.handle (interp_ssa (interp_invalid t sorry) env)

/-
### Interpretation layers for Toy programs

We first interpret away nonexsting invalid operations, then the memory layer,
and finally the Toy operations themselves.
-/

#check interp_invalid
#check interp_ssa
#check interp_toy


/-
### Examples and testing
-/

-- The following extends #reduce with a (skipProofs := true/false) parameter
-- to not reduce proofs in the kernel. Reducing proofs would cause constant
-- timeouts, and proofs are used implicitly through well-founded induction for
-- mutual definitions.
-- See: https://leanprover.zulipchat.com/#narrow/stream/270676-lean4/topic/Repr.20instance.20for.20functions/near/276504682

open Lean
open Lean.Parser.Term
open Lean.Elab.Command
open Lean.Elab
open Lean.Meta

elab "#reduce " skipProofs:group(atomic("(" &"skipProofs") " := " (trueVal <|> falseVal) ")") term:term : command =>
  let skipProofs := skipProofs[3].isOfKind ``trueVal
  withoutModifyingEnv <| runTermElabM (some `_check) fun _ => do
    -- dbg_trace term
    let e ← Term.elabTerm term none
    Term.synthesizeSyntheticMVarsNoPostponing
    let (e, _) ← Term.levelMVarToParam (← instantiateMVars e)
    withTheReader Core.Context (fun ctx => { ctx with options := ctx.options.setBool `smartUnfolding false }) do
      let e ← withTransparency (mode := TransparencyMode.all) <| reduce e (skipProofs := skipProofs) (skipTypes := false)
      logInfo e

---

def transpose_stmt := [mlir_bb_stmt|
  %t2 = "toy.transpose"(%t1): tensor<2×4×i32> -> tensor<4×2×i32>
]

def constant_stmt := [mlir_bb_stmt|
  %t = "toy.constant"() {value=dense<[[1,2],[3,4]]>: tensor<2×2×i32>}:
    () -> tensor<2×2×i32>
]

#reduce constant_stmt

def double_transpose := [mlir_bb|
  ^dbl:
    %t2 = "toy.transpose"(%t1): tensor<2×4×i32> -> tensor<4×2×i32>
    %t3 = "toy.transpose"(%t2): tensor<4×2×i32> -> tensor<2×4×i32>
]

#reduce (skipProofs := true)
  toy_semantics_bb double_transpose

#reduce (skipProofs := true)
  run_toy (toy_semantics_bbstmt transpose_stmt) [[]]

#reduce (skipProofs := true)
  run_toy (toy_semantics_bbstmt constant_stmt) [[]]

theorem double_transpose_correct:
  ∀ (t1: RankedTensor Int [2,4]),
    run_toy (toy_semantics_bb double_transpose)
      [[("t1", ⟨MLIRTy.tensorRanked [2,4] (MLIRTy.int 32), t1⟩)]]
    =
    Fitree.ret ((), [[
      (SSAVal.SSAVal "t1", ⟨MLIRTy.tensorRanked [2,4] (MLIRTy.int 32), t1⟩),
      (SSAVal.SSAVal "t2", ⟨MLIRTy.tensorRanked [4,2] (MLIRTy.int 32),
                           transpose t1⟩),
      (SSAVal.SSAVal "t3", ⟨MLIRTy.tensorRanked [2,4] (MLIRTy.int 32), t1⟩)
    ]]) := by
  intros t1
  unfold double_transpose
  simp
  simp [double_transpose, toy_semantics_bb, toy_semantics_bbstmt]; simp_itree
  simp [interp_invalid]; simp_itree
  simp [interp_ssa]; simp_itree
  rw [transpose_involutive]
  rfl
