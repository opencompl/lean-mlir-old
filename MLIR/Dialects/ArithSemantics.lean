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

def arith_semantics_op: IOp Δ →
    Option (Fitree (RegionE Δ +' UBE +' ArithE) (BlockResult Δ))

  | IOp.mk "arith.constant" [] [] 0 attrs (.fn (.tuple []) τ₁) => some <|
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
      | _ => do
            Fitree.trigger $ UBE.DebugUB "non maching type of arith.const"
            return BlockResult.Ret []

  | IOp.mk "arith.cmpi" [ ⟨(.int sgn sz), lhs⟩, ⟨(.int sgn' sz'), rhs⟩ ] [] 0
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
            | _ => do
                Fitree.trigger $ UBE.DebugUB "unable to find predicate"
                return BlockResult.Ret []
      else do
        Fitree.trigger $ UBE.DebugUB "lhs, rhs, unequal sizes (cmp)"
        return BlockResult.Ret []

  | IOp.mk "arith.addi" [⟨.int sgn sz, lhs⟩, ⟨ .int sgn' sz', rhs⟩] [] 0 _ _ => some do
      if EQ: sgn = sgn' /\ sz = sz' then
          have rhs': (MLIRType.int sgn sz).eval := by (
            simp [EQ.1, EQ.2];
            exact rhs)
          let r ← Fitree.trigger (ArithE.AddI sz lhs rhs')
          return BlockResult.Next ⟨.int sgn sz, r⟩
      else
        Fitree.trigger $ UBE.DebugUB "lhs, rhs, unequal sizes (add)"
        return BlockResult.Ret []
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
  | CmpI sz pred lhs rhs =>
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

instance: Semantics arith where
  E := ArithE
  semantics_op := arith_semantics_op
  handle := ArithE.handle

/-
### Basic examples
-/

private def cst1: BasicBlock arith := [mlir_bb|
  ^bb:
    %true = "arith.constant" () {value = 1: i1}: () -> i1
    %false = "arith.constant" () {value = 0: i1}: () -> i1
    %r1 = "arith.constant" () {value = 25: i32}: () -> i32
    %r2 = "arith.constant" () {value = 17: i32}: () -> i32
    %r = "arith.addi" (%r1, %r2): (i32, i32) -> i32
    %b1 = "arith.cmpi" (%r, %r1) {predicate = 5 /- sge -/}: (i32, i32) -> i1
    %b2 = "arith.cmpi" (%r2, %r) {predicate = 8 /- ugt -/}: (i32, i32) -> i1
]

#eval run (Δ := arith) ⟦cst1⟧ (SSAEnv.empty (δ := arith))


/-
### Theorems
-/

private def add1: BasicBlockStmt arith := [mlir_bb_stmt|
  %r = "arith.addi"(%n, %m): (i32, i32) -> i32
]
private def add2: BasicBlockStmt arith := [mlir_bb_stmt|
  %r = "arith.addi"(%m, %n): (i32, i32) -> i32
]

theorem add_commutative:
  forall (n m: FinInt 32),
    run ⟦add1⟧ [[ ("n", ⟨.i32, n⟩), ("m", ⟨.i32, m⟩) ]] =
    run ⟦add2⟧ [[ ("n", ⟨.i32, n⟩), ("m", ⟨.i32, m⟩) ]] := by
  intros n m
  simp [Denote.denote]
  simp [run, add1, add2, denoteBBStmt, denoteOp]
  simp [interp_ub]; simp_itree
  simp [interp_ssa]; simp_itree
  simp [Semantics.handle, ArithE.handle]; simp_itree
  simp [FinInt.add_comm]
