import MLIR.Dialects.ToyModel
import MLIR.Dialects.BuiltinModel
import MLIR.Semantics.Fitree
import MLIR.Semantics.Verifier
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.InvalidOp
import MLIR.Semantics.TensorElem
import MLIR.Util.Metagen
import MLIR.Util.Reduce

import MLIR.AST
import MLIR.EDSL

import Lean

open MLIR.AST


/- To be automatically generated -/

inductive ToyOp: Type → Type :=
  | Constant:
      (cst_D: DimList) → (cst_τ: MLIRTy) → (cst: TensorLiteral cst_D cst_τ) →
      ToyOp (RankedTensor cst_D cst_τ)
  | Transpose:
      (τ: MLIRTy) → (n m: Nat) →
      RankedTensor [Dimension.Known n, Dimension.Known m] τ →
      ToyOp (RankedTensor [Dimension.Known m, Dimension.Known n] τ)
  | Reshape:
      (τ: MLIRTy) → (D D': DimList) → (H: D.known) → (H': D'.known) →
      (Hprod: D'.prod = D.prod) →
      RankedTensor D τ →
      ToyOp (RankedTensor D' τ)

/- To be automatically generated (hopefully; basically this is the
   verification stuff) -/

def toy_semantics_op (ret_name: Option SSAVal) (op: Op builtin):
      Fitree (InvalidOpE builtin +' SSAEnvE builtin +' ToyOp) Unit :=

  match op with
  | Op.mk "toy.constant" [] [] [] attrs
        (.fn (.tuple []) (builtin.tensor D₁ τ₁)) =>
      match AttrDict.find attrs "value" with
      | some (builtin.dense_tensor_attr elem D₂ τ₂) =>
          match TensorLiteral.ofTensorElem elem D₁ τ₁ with
          | none =>
              Fitree.trigger (InvalidOpE.InvalidOp op)
          | some t_lit => do
              let t ← Fitree.trigger <| ToyOp.Constant D₁ τ₁ t_lit
              SSAEnv.set? (builtin.tensor D₁ τ₁) ret_name t
      | _ =>
          Fitree.trigger (InvalidOpE.InvalidOp op)

  | Op.mk "toy.transpose" [t_name] [] [] _ (.fn (builtin.tensor D τ) τ₂) =>
      match D with
      | [Dimension.Known n, Dimension.Known m] => do
          let t ← Fitree.trigger (SSAEnvE.Get (builtin.tensor
                  [Dimension.Known n, Dimension.Known m] τ) t_name);
          let t' ← Fitree.trigger (ToyOp.Transpose τ n m t);
          SSAEnv.set? (builtin.tensor [Dimension.Known m, Dimension.Known n] τ)
            ret_name t'
      | _ =>
          Fitree.trigger (InvalidOpE.InvalidOp op)

  | Op.mk "toy.reshape" [t_name] [] [] _
        (.fn (builtin.tensor D τ₁) (builtin.tensor D' τ₂)) =>
      if H: τ₁ = τ₂
        ∧ DimList.known D
        ∧ DimList.known D'
        ∧ DimList.prod D' = DimList.prod D then do
        let t ← Fitree.trigger (SSAEnvE.Get (builtin.tensor D τ₁) t_name);
        let t' ← Fitree.trigger (ToyOp.Reshape τ₁ D D'
                H.2.1 H.2.2.1 H.2.2.2 t);
        let t': RankedTensor D' τ₂ := cast (by rw [H.1]) t';
        SSAEnv.set? (builtin.tensor D' τ₂) ret_name t'
      else
        Fitree.trigger (InvalidOpE.InvalidOp op)

  | _ =>
      Fitree.trigger (InvalidOpE.InvalidOp op)

def toy_semantics_bbstmt:
      BasicBlockStmt builtin →
      Fitree (InvalidOpE builtin +' (SSAEnvE builtin) +' ToyOp) Unit
  | BasicBlockStmt.StmtAssign val _ op =>
      toy_semantics_op (some val) op
  | BasicBlockStmt.StmtOp op =>
      toy_semantics_op none op

-- TODO: toy_semantics_bb: handle basic block arguments
@[simp]
def toy_semantics_bb:
      BasicBlock builtin →
      Fitree (InvalidOpE builtin +' (SSAEnvE builtin) +' ToyOp) Unit
  | BasicBlock.mk name args [] =>
      Fitree.ret ()
  | BasicBlock.mk name args (op1::ops) =>
      List.foldr (fun t acc => Fitree.bind acc (fun _ => t))
                 (toy_semantics_bbstmt op1)
                 (ops.map toy_semantics_bbstmt)

/- Manually specified: ToyOp event handler -/

def ToyOp.handle {E}: ToyOp ~> Fitree E :=
  fun _ e => match e with
  | ToyOp.Constant D τ t_lit =>
      return RankedTensor.ofTensorLiteral t_lit
  | ToyOp.Transpose α n m t =>
      return transpose t
  | ToyOp.Reshape α D D' H H' Hprod t =>
      return reshape D' H H' Hprod t

-- Interpretation in context

def interp_toy {E} (t: Fitree (ToyOp +' E) R): Fitree E R :=
  interp (Fitree.case_ ToyOp.handle (fun T => @Fitree.trigger E E T _)) t

@[simp]
def run_toy (t: Fitree (InvalidOpE builtin +' SSAEnvE builtin +' ToyOp) Unit)
    (env: SSAEnv builtin): Fitree PVoid (Unit × SSAEnv builtin) :=
  interp ToyOp.handle (interp_ssa (interp_invalid! t) env)

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

-- TODO: Can we infer the builtin in there?
def transpose_stmt: BasicBlockStmt builtin := [mlir_bb_stmt|
  %t2 = "toy.transpose"(%t1): tensor<2×4×i32> -> tensor<4×2×i32>
]

def constant_stmt: BasicBlockStmt builtin := [mlir_bb_stmt|
  %t = "toy.constant"() {value=dense<[[1,2],[3,4]]>: tensor<2×2×i32>}:
    () -> tensor<2×2×i32>
]

#reduce constant_stmt

def double_transpose: BasicBlock builtin := [mlir_bb|
  ^dbl:
    %t2 = "toy.transpose"(%t1): tensor<2×4×i32> -> tensor<4×2×i32>
    %t3 = "toy.transpose"(%t2): tensor<4×2×i32> -> tensor<2×4×i32>
]

#reduce toy_semantics_bb double_transpose

#eval Fitree.run <| run_toy (toy_semantics_bbstmt transpose_stmt) [[]]

#eval Fitree.run <| run_toy (toy_semantics_bbstmt constant_stmt) [[]]

theorem double_transpose_correct:
  ∀ (t1: RankedTensor [.Known 2, .Known 4] (.int 32)),
    run_toy (toy_semantics_bb double_transpose)
      [[("t1", ⟨builtin.tensor [.Known 2, .Known 4] (.int 32), t1⟩)]]
    =
    Fitree.ret ((), [[
      (SSAVal.SSAVal "t1", ⟨builtin.tensor [.Known 2, .Known 4] (.int 32), t1⟩),
      (SSAVal.SSAVal "t2", ⟨builtin.tensor [.Known 4, .Known 2] (.int 32),
                           transpose t1⟩),
      (SSAVal.SSAVal "t3", ⟨builtin.tensor [.Known 2, .Known 4] (.int 32), t1⟩)
    ]]) := by
  intros t1
  unfold double_transpose
  simp
  simp [double_transpose, toy_semantics_bb, toy_semantics_bbstmt]; simp_itree
  simp [interp_invalid]; simp_itree
  simp [interp_ssa]; simp_itree
  rw [transpose_involutive]
