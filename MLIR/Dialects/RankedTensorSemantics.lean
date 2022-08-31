/-
## `rankedtensor` dialect

This file shows how to use `RankedTensor` which lives
in `MLIRTy`, and the corresponding theory of projections
that are necessary to manipulate these.

-/

import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
open MLIR.AST



instance rankedtensor: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

theorem eval_of_coe [δ₁: Dialect α₁ σ₁ ε₁] [δ₂: Dialect α₂ σ₂ ε₂] [c: CoeDialect δ₁ δ₂]
    (T: MLIRType δ₁): MLIRType.eval (coeMLIRType (δ₂ := δ₂) T) = MLIRType.eval T := sorry
#check MLIRType
-- @lephe: do you want me to thread the dialect projection everywhere?
-- I have a projection from Δ to builtin
def rankedtensor_semantics_op [P: DialectProjection Δ builtin]: IOp Δ →
      Option (Fitree (RegionE Δ +' UBE +' LinalgE) (BlockResult Δ))
  | IOp.mk "rankedtensor.dim" _ [⟨.extended sΔ, v⟩] _ _ attrs =>
        match AttrDict.find attrs "map" with
        | some (.affine affine_map?) => do
            match H: P.project_σ _ _ sΔ with
                | some (builtin.σ.tensor dims τ) => .some do
                  -- project_ε gets stuck unless it seems the proof of project_σ
                  let input: RankedTensor dims τ :=
                         cast (by rw [H]) (P.project_ε sΔ v)
                  /-
                     application type mismatch
                       { fst := MLIRType.extended sΔ, snd := input }
                     argument
                       input
                     has type
                       RankedTensor dims τ : Type
                     but is expected to have type
                       MLIRType.eval (MLIRType.extended sΔ) : Type
                  return (BlockResult.Ret [⟨.extended sΔ, input⟩])
                  -/
                  /-
                  application type mismatch
                    { fst := coeMLIRType (builtin.tensor dims τ), snd := input }
                  argument
                    input
                  has type
                    RankedTensor dims τ : Type
                  but is expected to have type
                    MLIRType.eval (coeMLIRType (builtin.tensor dims τ)) : Type
                  return BlockResult.Ret [⟨builtin.tensor dims τ, input⟩]
                  -/
                  /-
                  application type mismatch
                    { fst := coeMLIRType (builtin.tensor dims τ), snd := input }
                  argument
                    input
                  has type
                    RankedTensor dims τ : Type
                  but is expected to have type
                    MLIRType.eval (coeMLIRType (builtin.tensor dims τ)) : Type
                  return BlockResult.Ret [⟨builtin.tensor dims τ, input⟩]
                  -/
                  -- @lephe: do I really want to reason about this? >_<
                  return BlockResult.Ret [⟨builtin.tensor dims τ, by {
                      rewrite [eval_of_coe];
                      rewrite [MLIRType.eval];
                      simp [builtin.ε];
                      simp [builtin.tensor];
                      exact input;
                  } ⟩]
                 | _ => none
        | _ => none
  | _ => none

instance: Semantics rankedtensor where
  E := fun T => Void
  semantics_op := sorry -- rankedtensor_semantics_op
  handle T voidT := nomatch voidT