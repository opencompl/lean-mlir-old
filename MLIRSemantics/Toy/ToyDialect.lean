import MLIRSemantics.Toy.Toy
import MLIRSemantics.Fitree
import MLIRSemantics.Verifier
import MLIRSemantics.SSAEnv

import MLIR.AST
open MLIR.AST

/- [Automatically generated] -/

inductive ToyOp: Type → Type _ :=
  | Constant: (α: Type) → (n m: Nat) → Tensor α [n,m] →
      ToyOp (Tensor α [n,m])
  | Transpose: (α: Type) → (n m: Nat) → Tensor α [n,m] →
      ToyOp (Tensor α [m,n])
  | Reshape: (α: Type) → (n m n' m': Nat) → (H: n'*m' = n*m) → Tensor α [n,m] →
      ToyOp (Tensor α [n',m'])

/- Hopefully this could be automatically generated -/

def ToyOp.semantics: Op → Fitree (psum SSAEnvE ToyOp) Unit
  | Op.mk "toy.constant" [] [] [] attributes
        (MLIRTy.fn (MLIRTy.tuple []) _) =>
      return ()
  | Op.mk "toy.transpose" [t_name] [] [] _ (MLIRTy.fn
        (MLIRTy.tensor [Dimension.Known (n:Nat), Dimension.Known (m:Nat)] _) _) => do
      /- TODO: Still need to model MLIR tensors -/
--      let t ← Fitree.trigger (@SSAEnvE.Get (Tensor Int [n,m]) _ t_name);
--      let t' ← Fitree.trigger (ToyOp.Transpose Int n m t);
      return ()
  | _ =>
      return ()

/- Manually specified: ToyOp event handler -/
