/-
## `arith` dialect

This file formalises part of the `arith` dialect. The goal is to showcase
operations on multiple types (with overloading) and basic reasoning. `arith`
does not have new datatypes, but it supports operations on tensors and vectors,
which are some of the most complex builtin types.
-/

import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Dialects.BuiltinModel
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
open MLIR.AST

/-
### Dialect extensions

`arith` has no extended types or attributes.
-/

instance arith: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

/-
### Dialect operations

In order to support type overloads while keeping reasonably-strong typing on
operands and disallowing incorrect types in the operation arguments, we define
scalar, tensor, and vector overloads of each operation.
-/

inductive ArithE: Type → Type :=
  | AddI: (sz: Nat) → (lhs rhs: FinInt sz) →
          ArithE (FinInt sz)
  | AddT: (sz: Nat) → (D: DimList) → (lhs rhs: RankedTensor D (.int sgn sz)) →
          ArithE (RankedTensor D (.int sgn sz))
  | AddV: (sz: Nat) → (sc fx: List Nat) →
          (lhs rhs: Vector sc fx (.int sgn sz)) →
          ArithE (Vector sc fx (.int sgn sz))

def arith_semantics_op {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} (ret: Option SSAVal):
    Op Gδ → Option (Fitree (SSAEnvE Gδ +' ArithE) (BlockResult Gδ))

  | Op.mk "arith.addi" [lhs, rhs] [] [] _ (.fn (.tuple [τ₁, τ₂]) τ) =>
      if h: τ₁ = τ₂ ∧ τ₁ = τ then
        match τ with
        | .int sgn sz => some do
            let lhs ← Fitree.trigger (SSAEnvE.Get (δ := Gδ) (.int sgn sz) lhs)
            let rhs ← Fitree.trigger (SSAEnvE.Get (δ := Gδ) (.int sgn sz) rhs)
            let r ← Fitree.trigger (ArithE.AddI sz lhs rhs)
            SSAEnv.set? (δ := Gδ) (.int sgn sz) ret r
            return BlockResult.Next
        | _ => none
      else none

  | _ => none

def ArithE.handle {E}: ArithE ~> Fitree E := fun _ e =>
  match e with
  | AddI sz lhs rhs =>
      return (lhs + rhs)
  | AddT sz D lhs rhs =>
      -- TODO: Implementation of ArithE.AddT (tensor addition)
      return default
  | AddV sz sc fx lhs rhs =>
      -- TODO: Implementation of ArithE.AddV (vector addition)
      return default

instance: Semantics arith where
  E := ArithE
  semantics_op := arith_semantics_op
  handle := ArithE.handle

/-
### Theorems
-/

def add1: BasicBlockStmt arith := [mlir_bb_stmt|
  %r = "arith.addi"(%n, %m): (i32, i32) -> i32
]
def add2: BasicBlockStmt arith := [mlir_bb_stmt|
  %r = "arith.addi"(%m, %n): (i32, i32) -> i32
]

theorem add_commutative:
  forall (n m: FinInt 32),
    run ⟦add1⟧ [[ ("n", ⟨.i32, n⟩), ("m", ⟨.i32, m⟩) ]] =
    run ⟦add2⟧ [[ ("n", ⟨.i32, n⟩), ("m", ⟨.i32, m⟩) ]] := by
  intros n m
  simp [Denote.denote]
  simp [run, semantics_bbstmt, semantics_op!, Semantics.semantics_op]
  simp [arith_semantics_op, Semantics.handle, add1, add2]
  simp [interp_ub!]; simp_itree
  simp [interp_ssa]; simp_itree
  rw [FinInt.add_comm]
