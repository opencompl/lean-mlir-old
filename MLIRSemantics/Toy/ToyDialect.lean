import MLIRSemantics.Toy.Toy
import MLIRSemantics.Fitree
import MLIRSemantics.Verifier
import MLIRSemantics.SSAEnv
import MLIRSemantics.InvalidOp

import MLIR.AST
import MLIR.EDSL

import Lean

open MLIR.AST


/- To be automatically generated -/

inductive ToyOp: Type → Type _ :=
  | Constant: (α: Type) → (n m: Nat) →
      RankedTensor α [Dimension.Known n, Dimension.Known m] →
      ToyOp (RankedTensor α [Dimension.Known n, Dimension.Known m])
  | Transpose: (α: Type) → (n m: Nat) →
      RankedTensor α [Dimension.Known n, Dimension.Known m] →
      ToyOp (RankedTensor α [Dimension.Known m, Dimension.Known n])
  | Reshape: (α: Type) → (D D': DimList) → (H: D.known) → (H': D'.known) →
      (Hprod: D'.prod = D.prod) →
      RankedTensor α D →
      ToyOp (RankedTensor α D')

/- To be automatically generated -/

def toy_semantics_op (ret_name: Option SSAVal):
      Op → Fitree (InvalidOpE +' SSAEnvE +' ToyOp) Unit

  | Op.mk "toy.constant" [] [] [] attributes (MLIRTy.fn (MLIRTy.tuple []) _) =>
      -- TODO: Access attributes
      return ()

  | Op.mk "toy.transpose" [t_name] [] [] _ (MLIRTy.fn τ₁ τ₂) =>
      match τ₁ with
      | MLIRTy.tensor [Dimension.Known n, Dimension.Known m] τ => do
          let t ← Fitree.trigger (@SSAEnvE.Get (MLIRTy.tensor [Dimension.Known
                  n, Dimension.Known m] τ) _ t_name);
          let t' ← Fitree.trigger (ToyOp.Transpose τ.eval n m t);
          match ret_name with
          | some name =>
              Fitree.trigger (@SSAEnvE.Set (MLIRTy.tensor [Dimension.Known m,
                Dimension.Known n] τ) name t')
          | none =>
              return ()
      | _ =>
          Fitree.trigger InvalidOpE.InvalidOp

  | Op.mk "toy.reshape" [t_name] [] [] _ (MLIRTy.fn τ₁ τ₂) =>
      match τ₁, τ₂ with
      | MLIRTy.tensor D σ₁, MLIRTy.tensor D' σ₂ =>
          if H: σ₁ = σ₂
             ∧ DimList.known D
             ∧ DimList.known D'
             ∧ DimList.prod D' = DimList.prod D then do
            let t ←Fitree.trigger (@SSAEnvE.Get (MLIRTy.tensor D σ₁) _ t_name);
            let t' ← Fitree.trigger (ToyOp.Reshape σ₁.eval D D'
                     H.2.1 H.2.2.1 H.2.2.2 t);
            let t': RankedTensor σ₂.eval D' := cast (by rw [H.1]) t';
            match ret_name with
            | some name =>
                Fitree.trigger (@SSAEnvE.Set (MLIRTy.tensor D' σ₂) name t')
            | none =>
                return ()
          else
            Fitree.trigger InvalidOpE.InvalidOp
      | _, _ =>
          Fitree.trigger InvalidOpE.InvalidOp

  | _ =>
      Fitree.trigger InvalidOpE.InvalidOp

def toy_semantics_bbstmt:
      BasicBlockStmt → Fitree (InvalidOpE +' SSAEnvE +' ToyOp) Unit
  | BasicBlockStmt.StmtAssign val op =>
      toy_semantics_op (some val) op
  | BasicBlockStmt.StmtOp op =>
      toy_semantics_op none op

/- Manually specified: ToyOp event handler -/

def ToyOp.handle {E}: ToyOp ~> Fitree E :=
  fun _ e => match e with
  | ToyOp.Constant α n m t =>
      return t
  | ToyOp.Transpose α n m t =>
      return transpose t
  | ToyOp.Reshape α D D' H H' Hprod t =>
      return reshape D' H H' Hprod t

-- Interpretation in context

def interp_toy {E} (t: Fitree (ToyOp +' E) R): Fitree E R :=
  interp (case_ ToyOp.handle (fun T => @Fitree.trigger E E T _)) t

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
    -- TODO: add options or notation for setting the following parameters
    withTheReader Core.Context (fun ctx => { ctx with options := ctx.options.setBool `smartUnfolding false }) do
      let e ← withTransparency (mode := TransparencyMode.all) <| reduce e (skipProofs := skipProofs) (skipTypes := false)
      logInfo e

def transpose_stmt := [mlir_bb_stmt|
    %t2 = "toy.transpose"(%t1): tensor<2×4:i32> -> tensor<4×2:i32>
]

#reduce (skipProofs := true)
  toy_semantics_bbstmt transpose_stmt

#reduce (skipProofs := true)
  interp_invalid (toy_semantics_bbstmt transpose_stmt) _

#reduce (skipProofs := true)
  interp_ssa (interp_invalid (toy_semantics_bbstmt transpose_stmt) _) [[]]

#reduce (skipProofs := true)
  interp ToyOp.handle (interp_ssa (interp_invalid (toy_semantics_bbstmt transpose_stmt) _) [[]])
