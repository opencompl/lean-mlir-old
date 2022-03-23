import MLIRSemantics.Toy.Toy
import MLIRSemantics.Fitree
import MLIRSemantics.Verifier
import MLIRSemantics.SSAEnv

import MLIR.AST
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
      (Hprod: D.prod = D'.prod) →
      RankedTensor α D →
      ToyOp (RankedTensor α D')

/- To be automatically generated -/

def toy_semantics: Op → Fitree (psum SSAEnvE ToyOp) Unit
  | Op.mk "toy.constant" [] [] [] attributes (MLIRTy.fn (MLIRTy.tuple []) _) =>
      -- TODO: Access attributes
      return ()
  | Op.mk "toy.transpose" [t_name] [] [] _ (MLIRTy.fn τ₁ τ₂) =>
      match τ₁ with
      | MLIRTy.tensor [Dimension.Known n, Dimension.Known m] τ => do
          let t ← Fitree.trigger (@SSAEnvE.Get (MLIRTy.tensor [Dimension.Known
                  n, Dimension.Known m] τ) _ t_name);
          let t' ← Fitree.trigger (ToyOp.Transpose τ.eval n m t);
          -- TODO: Have proper return names (ie. match above Op)
          Fitree.trigger (@SSAEnvE.Set (MLIRTy.tensor [Dimension.Known m,
            Dimension.Known n] τ) "%ret" t')
      | _ =>
          return ()
  | Op.mk "toy.reshape" [t_name] [] [] _ (MLIRTy.fn τ₁ τ₂) =>
      match τ₁, τ₂ with
      | MLIRTy.tensor D σ₁, MLIRTy.tensor D' σ₂ =>
          if σ₁ = σ₂ then do
            let t ← Fitree.trigger (@SSAEnvE.Get (MLIRTy.tensor D σ₁) _ t_name);
            -- TODO: Propagate verification invariants!
            let t' ← Fitree.trigger (ToyOp.Reshape σ₁.eval D D'
                     sorry sorry sorry sorry);
            -- TODO: Cast σ₁ to σ₂?
            Fitree.trigger (@SSAEnvE.Set (MLIRTy.tensor D' σ₁) "%ret" t')
          else
            return ()
      | _, _ =>
          return ()
  | _ =>
      return () -- TODO: Maybe raise invalid instruction event for clarity?

/- Manually specified: ToyOp event handler -/
