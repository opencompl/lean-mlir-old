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
  | Zext: (sz₁: Nat) → (sz₂: Nat) → (FinInt sz₁) →
          ArithE (FinInt sz₂)
  | Select: (sz: Nat) → (b: FinInt 1) → (lhs rhs: FinInt sz) →
          ArithE (FinInt sz)

def unary_semantics_op (op: IOp Δ)
      (ctor: (sz: Nat) → FinInt sz → ArithE (FinInt sz)):
    Option (Fitree (RegionE Δ +' UBE +' ArithE) (BlockResult Δ)) :=
  match op with
  | IOp.mk _ [⟨.int sgn sz, arg⟩] [] 0 _ _ => some do
      let r ← Fitree.trigger (ctor sz arg)
      return BlockResult.Next ⟨.int sgn sz, r⟩
  | IOp.mk _ _ _ _ _ _ => none

def binary_semantics_op {Δ: Dialect α' σ' ε'}
      (name: String) (args: List ((τ: MLIRType Δ) × τ.eval))
      (ctor: (sz: Nat) → FinInt sz → FinInt sz → ArithE (FinInt sz)):
    Option (Fitree (RegionE Δ +' UBE +' ArithE) (BlockResult Δ)) :=
  match args with
  | [⟨.int sgn sz, lhs⟩, ⟨.int sgn' sz', rhs⟩] => some do
      if EQ: sgn = sgn' /\ sz = sz' then
        let r ← Fitree.trigger (ctor sz lhs (EQ.2 ▸ rhs))
        return BlockResult.Next ⟨.int sgn sz, r⟩
      else
        Fitree.trigger <| UBE.DebugUB s!"{name}: incompatible operand types"
        return BlockResult.Ret []
  | _ => none

def arith_semantics_op (o: IOp Δ):
    Option (Fitree (RegionE Δ +' UBE +' ArithE) (BlockResult Δ)) :=
  match o with
  | IOp.mk "constant" [] [] 0 attrs (.fn (.tuple []) τ₁) =>
      match AttrDict.find attrs "value" with
      | some (.int value τ₂) =>
          if τ₁ = τ₂ then
            match τ₂ with
            | .int sgn sz => some do
                -- TODO: Check range of constants
                let v := FinInt.ofInt sz value
                return BlockResult.Next ⟨.int sgn sz, v⟩
            | _ => none
          else none
      | some _
      | none => none

  | IOp.mk "cmpi" [ ⟨(.int sgn sz), lhs⟩, ⟨(.int sgn' sz'), rhs⟩ ] [] 0
    attrs _ =>
      if EQ: sgn = sgn' /\ sz = sz' then
            match attrs.find "predicate" with
            | some (.int n (.int .Signless 64)) =>
                match (ComparisonPred.ofInt n) with
                | some pred => some do
                  let r ← Fitree.trigger (ArithE.CmpI sz pred lhs (EQ.2 ▸ rhs))
                  return BlockResult.Next ⟨.i1, r⟩
                | none => none
            | some _
            | none => none
      else none

  | IOp.mk "zext" [⟨.int sgn₁ sz₁, value⟩] [] 0 _
    (.fn (.tuple [.int _ _]) (.int sgn₂ sz₂)) =>
      if sgn₁ = sgn₂ then some do
        let r ← Fitree.trigger (ArithE.Zext sz₁ sz₂ value)
        return BlockResult.Next ⟨.int sgn₂ sz₂, r⟩
      else none

  | IOp.mk "select" [⟨.i1, b⟩, ⟨.int sgn sz, lhs⟩, ⟨.int sgn' sz', rhs⟩] [] 0
    _ _ =>
      if EQ: sgn = sgn' /\ sz = sz' then some do
        let r ← Fitree.trigger (ArithE.Select sz b lhs (EQ.2 ▸ rhs))
        return BlockResult.Next ⟨.int sgn sz, r⟩
      else none

  | IOp.mk "negi" _ _ _ _ _ =>
      unary_semantics_op o ArithE.NegI
  | IOp.mk name args _ _ _ _ =>
      if name = "addi" then
        binary_semantics_op name args ArithE.AddI
      else if name = "subi" then
        binary_semantics_op name args ArithE.SubI
      else if name = "andi" then
        binary_semantics_op name args ArithE.AndI
      else if name = "ori" then
        binary_semantics_op name args ArithE.OrI
      else if name = "xori" then
        binary_semantics_op name args ArithE.XorI
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
      return FinInt.ofInt 1 (if b then 1 else 0)
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
  | Zext _ sz₂ value =>
      return FinInt.zext sz₂ value
  | Select _ b lhs rhs =>
      return if b.toUint = 1 then lhs else rhs

instance arithSemantics : Semantics arith where
  E := ArithE
  semantics_op := arith_semantics_op
  handle := ArithE.handle

/-
### Semantics of individual operations

In principle we would compute the semantics of entire programs simply by
unfolding the definitions. But simp and dsimp have many problems which makes
this extremely slow, buggy, or infeasible even for programs with only a couple
of operations. We work around this issue by precomputing the semantics of
individual operations and then substituting them as needed.
-/

private abbrev ops.constant (output: SSAVal) (value: Int):
    BasicBlockStmt arith :=
  .StmtAssign output none <|
    .mk "constant" [] [] [] (.mk [.mk "value" (.int value .i32)]) (.fn (.tuple []) .i32)

private abbrev ops.negi (output input: SSAVal): BasicBlockStmt arith :=
  .StmtAssign output none <|
    .mk "negi" [input] [] [] (.mk []) (.fn (.tuple [.i32]) .i32)

private abbrev ops._binary (name: String) (output lhs rhs: SSAVal):
    BasicBlockStmt arith :=
  .StmtAssign output none <|
    .mk name [lhs, rhs] [] [] (.mk []) (.fn (.tuple [.i32, .i32]) .i32)

private abbrev ops.addi := ops._binary "addi"
private abbrev ops.subi := ops._binary "subi"
private abbrev ops.andi := ops._binary "andi"
private abbrev ops.ori  := ops._binary "ori"
private abbrev ops.xori := ops._binary "xori"

private theorem ops.constant.sem output value:
    denoteBBStmt arith (ops.constant output value) =
  Fitree.Vis (E := UBE +' SSAEnvE arith +' Semantics.E arith)
    (Sum.inr <| Sum.inl <| SSAEnvE.Set .i32 output (FinInt.ofInt 32 value)) fun _ =>
  Fitree.ret (BlockResult.Next (δ := arith)
    ⟨.i32, FinInt.ofInt 32 value⟩) := by
  simp [ops.constant, denoteBBStmt, denoteOp, Semantics.semantics_op]
  simp_itree
  simp [arith_semantics_op, AttrDict.find, List.find?, AttrEntry.key, AttrEntry.value]
  simp_itree

private theorem ops.negi.sem output input:
    denoteBBStmt arith (ops.negi output input) =
  Fitree.Vis (E := UBE +' SSAEnvE arith +' Semantics.E arith)
    (Sum.inr <| Sum.inl <| SSAEnvE.Get .i32 input) fun r =>
  Fitree.Vis (Sum.inr <| Sum.inr <| ArithE.NegI 32 r) fun r =>
  Fitree.Vis (Sum.inr <| Sum.inl <| SSAEnvE.Set .i32 output r) fun _ =>
  Fitree.ret (BlockResult.Next ⟨.i32, r⟩) := by
  simp [ops.negi, denoteBBStmt, denoteOp, Semantics.semantics_op]
  simp_itree

private theorem ops._binary.sem name ctor output lhs rhs:
    (forall (n m: FinInt 32),
      arith_semantics_op (Δ := arith)
        (IOp.mk name [⟨.i32, n⟩, ⟨.i32, m⟩] [] 0 (.mk []) (.fn (.tuple [.i32, .i32]) .i32)) =
      binary_semantics_op name [⟨.i32, n⟩, ⟨.i32, m⟩] ctor) →
    denoteBBStmt arith (ops._binary name output lhs rhs) =
  Fitree.Vis (E := UBE +' SSAEnvE arith +' Semantics.E arith)
    (Sum.inr <| Sum.inl <| SSAEnvE.Get .i32 lhs) fun lhs =>
  Fitree.Vis (Sum.inr <| Sum.inl <| SSAEnvE.Get .i32 rhs) fun rhs =>
  Fitree.Vis (Sum.inr <| Sum.inr <| ctor 32 lhs rhs) fun r =>
  Fitree.Vis (Sum.inr <| Sum.inl <| SSAEnvE.Set .i32 output r) fun _ =>
  Fitree.ret (BlockResult.Next ⟨.i32, r⟩) := by
  intro h
  simp [denoteBBStmt, denoteOp, Semantics.semantics_op]
  simp_itree
  simp [h, binary_semantics_op]
  simp_itree

private abbrev ops.addi.sem output lhs rhs :=
  ops._binary.sem "addi" ArithE.AddI output lhs rhs (fun _ _ => rfl)
private abbrev ops.subi.sem output lhs rhs :=
  ops._binary.sem "subi" ArithE.SubI output lhs rhs (fun _ _ => rfl)
private abbrev ops.andi.sem output lhs rhs :=
  ops._binary.sem "andi" ArithE.AndI output lhs rhs (fun _ _ => rfl)
private abbrev ops.ori.sem output lhs rhs :=
  ops._binary.sem "ori" ArithE.OrI output lhs rhs (fun _ _ => rfl)
private abbrev ops.xori.sem output lhs rhs :=
  ops._binary.sem "xori" ArithE.XorI output lhs rhs (fun _ _ => rfl)

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

#eval run (Δ := arith) ⟦cst1⟧ (SSAEnv.empty (δ := arith))


/-
### Rewriting heorems
-/

/- Commutativity of addition -/

namespace th1
def LHS: BasicBlockStmt arith := [mlir_bb_stmt|
  %r = "addi"(%n, %m): (i32, i32) -> i32
]
def RHS: BasicBlockStmt arith := [mlir_bb_stmt|
  %r = "addi"(%m, %n): (i32, i32) -> i32
]

theorem equivalent (n m: FinInt 32):
    run ⟦LHS⟧ (SSAEnv.One [ ("n", ⟨.i32, n⟩), ("m", ⟨.i32, m⟩) ]) =
    run ⟦RHS⟧ (SSAEnv.One [ ("n", ⟨.i32, n⟩), ("m", ⟨.i32, m⟩) ]) := by
  simp [Denote.denote]
  simp [run, LHS, RHS, denoteBBStmt, denoteOp]
  simp [interp_ub, SSAEnv.get]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [Semantics.handle, ArithE.handle, SSAEnv.get]; simp_itree
  simp [FinInt.add_comm]
end th1

/- LLVM InstCombine: `C-(X+C2) --> (C-C2)-X`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L1794 -/

theorem FinInt.sub_add_dist: forall (C X C2: FinInt sz),
    C - (X + C2) = (C - C2) - X := by
  intros C X C2
  apply eq_of_toUint_cong2
  simp [cong2, FinInt.sub_toUint, FinInt.add_toUint]
  apply FinInt.mod2_fequal
  simp [Int.sub_add_dist, Int.sub_assoc]

namespace th2
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %t = "addi"(%X, %C2): (i32, i32) -> i32
    %r = "subi"(%C, %t): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %t = "subi"(%C, %C2): (i32, i32) -> i32
    %r = "subi"(%t, %X): (i32, i32) -> i32
]
def INPUT (C X C2: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("C", ⟨.i32, C⟩), ("X", ⟨.i32, X⟩), ("C2", ⟨.i32, C2⟩)
]

theorem equivalent (C X C2: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT C X C2) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT C X C2) |>.snd.get "r" .i32) := by
  simp [LHS, RHS, INPUT]
  simp [run, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp]; simp_itree
  simp [interp_ub]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  apply FinInt.sub_add_dist
end th2

/- LLVM InstCombine: `~X + C --> (C-1) - X`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L882 -/

theorem FinInt.comp_add: sz > 0 → forall (X C: FinInt sz),
    (X ^^^ -1) + C = (C - 1) - X := by
  intros h_sz X C
  simp [←FinInt.comp_eq_xor_minusOne]
  apply eq_of_toUint_cong2
  simp [cong2, FinInt.add_toUint, FinInt.comp_toUint, FinInt.sub_toUint]
  simp [toUint_ofNat]
  have h: Int.ofNat 1 = 1 := by decide
  simp [h, mod2_idem ⟨by decide, Int.one_lt_two_pow h_sz⟩]
  simp [Int.sub_eq_add_neg, Int.add_assoc, FinInt.mod2_add_left]
  rw [←@Int.add_assoc _ (-1) _, @Int.add_comm _ (-1)]
  simp [@Int.add_comm (-X.toUint), Int.add_assoc]

namespace th3
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "constant"() {value = 1: i32}: () -> i32
    %_2 = "negi"(%_1): (i32) -> i32
    %_3 = "xori"(%X, %_2): (i32, i32) -> i32
    %r = "addi"(%_3, %C): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %o = "constant"() {value = 1: i32}: () -> i32
    %t = "subi"(%C, %o): (i32, i32) -> i32
    %r = "subi"(%t, %X): (i32, i32) -> i32
]
def INPUT (C X: FinInt 32): SSAEnv arith := SSAEnv.One [
    ("C", ⟨.i32, C⟩), ("X", ⟨.i32, X⟩)
]

theorem LHS.sem (C X: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT C X) |>.snd.get "r" .i32) =
      ((X ^^^ -1) + C: FinInt 32) := by
  simp [INPUT, LHS, run, denoteBBStmts, denoteBB]
  rw [ops.constant.sem]
  rw [ops.negi.sem]
  rw [ops.xori.sem]
  rw [ops.addi.sem]
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle]; dsimp_itree
  repeat (simp [SSAEnv.get, SSAEnv.set]; dsimp_itree)
  rfl

theorem RHS.sem (C X: FinInt 32):
    (run (denoteBB _ RHS []) (INPUT C X) |>.snd.get "r" .i32) =
      (C - 1 - X: FinInt 32) := by
  simp [INPUT, RHS, run, denoteBBStmts, denoteBB]
  rw [ops.constant.sem]
  rw [ops.subi.sem]
  rw [ops.subi.sem]
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle]; dsimp_itree
  repeat (simp [SSAEnv.get]; dsimp_itree)
  rfl

theorem equivalent (C X: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT C X) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT C X) |>.snd.get "r" .i32) := by
  rw [LHS.sem, RHS.sem]; simp
  apply FinInt.comp_add (by decide)
end th3

/- LLVM InstCombine: `-A + -B --> -(A + B)`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L1316 -/

theorem FinInt.neg_add_dist (A B: FinInt sz):
    -(A + B) = -A + -B := by
  apply eq_of_toUint_cong2
  simp [cong2, neg_toUint, add_toUint]
  apply mod2_fequal
  simp [Int.neg_add]

namespace th4
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "negi"(%A): (i32) -> i32
    %_2 = "negi"(%B): (i32) -> i32
    %r = "addi"(%_1, %_2): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "addi"(%A, %B): (i32, i32) -> i32
    %r = "negi"(%_1): (i32) -> i32
]
def INPUT (A B: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("A", ⟨.i32, A⟩), ("B", ⟨.i32, B⟩)
]

theorem equivalent (A B: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT A B) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT A B) |>.snd.get "r" .i32) := by
  simp [LHS, RHS, INPUT]
  simp [run, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp]; simp_itree
  simp [interp_ub]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  apply Eq.symm
  apply FinInt.neg_add_dist
end th4

/- LLVM InstCombine: `-(X - Y) --> (Y - X)`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L2290 -/

theorem FinInt.neg_sub_dist (X Y: FinInt sz):
    -(X - Y) = Y - X := by
  apply eq_of_toUint_cong2
  simp [cong2, neg_toUint, sub_toUint]
  apply mod2_fequal
  simp [Int.neg_sub]

namespace th5
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "subi"(%X, %Y): (i32, i32) -> i32
    %r = "negi"(%_1): (i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %r = "subi"(%Y, %X): (i32, i32) -> i32
]
def INPUT (X Y: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("X", ⟨.i32, X⟩), ("Y", ⟨.i32, Y⟩)
]

theorem equivalent (X Y: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT X Y) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT X Y) |>.snd.get "r" .i32) := by
  simp [LHS, RHS, INPUT]
  simp [run, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp]; simp_itree
  simp [interp_ub]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  simp [SSAEnv.get]; simp_itree
  apply FinInt.neg_sub_dist
end th5

/- LLVM InstCombine: `(A + 1) + ~B --> A - B`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L1331 -/

theorem FinInt.plus_one_plus_comp (A B: FinInt sz):
    (A + 1) + (B ^^^ -1) = A - B := by
  simp [←FinInt.comp_eq_xor_minusOne]
  apply eq_of_toUint_cong2
  simp [cong2, neg_toUint, add_toUint, sub_toUint, comp_toUint]
  simp [toUint_ofNat, (by decide: Int.ofNat 1 = 1)]
  -- Rearranging terms without the powerful Mathlib tactics is quite tedious
  simp [Int.add_comm _ (mod2 _ _), Int.add_assoc]
  simp [Int.sub_eq_add_neg]
  simp [←@Int.add_assoc A.toUint _, Int.add_comm A.toUint _]
  simp [←@Int.add_assoc 1 _, Int.add_comm 1 _]
  simp [Int.add_assoc, mod2_add_left]
  simp [Int.add_comm _ 1, ←Int.add_assoc _ 1 _, Int.add_assoc 1 _ _]
  simp [←Int.add_assoc, Int.add_right_neg, Int.zero_add]

namespace th6
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "constant"() {value = 1: i32}: () -> i32
    %_2 = "addi"(%A, %_1): (i32, i32) -> i32
    %_3 = "negi"(%_1): (i32) -> i32
    %_4 = "xori"(%B, %_3): (i32, i32) -> i32
    %r = "addi"(%_2, %_4): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %r = "subi"(%A, %B): (i32, i32) -> i32
]
def INPUT (A B: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("A", ⟨.i32, A⟩), ("B", ⟨.i32, B⟩)
]

theorem LHS.sem (A B: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT A B) |>.snd.get "r" .i32) =
      ((A + 1) + (B ^^^ -1): FinInt 32) := by
  simp [INPUT, LHS, run, denoteBBStmts, denoteBB]
  rw [ops.constant.sem]
  rw [ops.addi.sem]
  rw [ops.negi.sem]
  rw [ops.xori.sem]
  rw [ops.addi.sem]
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle]; dsimp_itree
  repeat (simp [SSAEnv.get, SSAEnv.set]; dsimp_itree)
  rfl

theorem RHS.sem (A B: FinInt 32):
    (run (denoteBB _ RHS []) (INPUT A B) |>.snd.get "r" .i32) =
      (A - B: FinInt 32) := by
  simp [INPUT, RHS, run, denoteBB, denoteBBStmts]
  rw [ops.subi.sem]
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle]; dsimp_itree
  repeat (simp [SSAEnv.get, SSAEnv.set]; dsimp_itree)

theorem equivalent (A B: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT A B) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT A B) |>.snd.get "r" .i32) := by
  rw [LHS.sem, RHS.sem]; simp
  apply FinInt.plus_one_plus_comp
end th6

/- LLVM InstCombine: `(~X) - (~Y) --> Y - X`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L1867 -/

theorem FinInt.comp_sub_comp (X Y: FinInt sz):
    (X ^^^ -1) - (Y ^^^ -1) = Y - X := by
  simp [←FinInt.comp_eq_xor_minusOne]
  apply eq_of_toUint_cong2
  simp [cong2, sub_toUint, comp_toUint]
  simp [Int.sub_eq_add_neg, Int.add_assoc, Int.neg_add, Int.neg_neg]
  simp [mod2_add_left]
  simp [←Int.add_assoc]
  rw [Int.add_comm _ (-_)]
  have h x: mod2 (-(2^sz) + x) sz = mod2 x sz := by sorry_arith
  simp [Int.add_assoc, h]
  rw [←Int.add_assoc (-1) _ _, Int.add_comm (-1) _]
  rw [Int.add_assoc _ (-1) _, ←Int.add_assoc _ 1 _]
  rw [Int.add_left_neg, Int.zero_add, Int.add_comm]

namespace th7
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "constant"() {value = 1: i32}: () -> i32
    %_2 = "negi"(%_1): (i32) -> i32
    %_3 = "xori"(%X, %_2): (i32, i32) -> i32
    %_4 = "xori"(%Y, %_2): (i32, i32) -> i32
    %r = "subi"(%_3, %_4): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %r = "subi"(%Y, %X): (i32, i32) -> i32
]
def INPUT (X Y: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("X", ⟨.i32, X⟩), ("Y", ⟨.i32, Y⟩)
]

theorem LHS.sem (X Y: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT X Y) |>.snd.get "r" .i32) =
      ((X ^^^ -1) - (Y ^^^ -1): FinInt 32) := by
  simp [INPUT, LHS, run, denoteBB, denoteBBStmts]
  rw [ops.constant.sem]
  rw [ops.negi.sem]
  rw [ops.xori.sem]
  rw [ops.xori.sem]
  rw [ops.subi.sem]
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle]; dsimp_itree
  repeat (simp [SSAEnv.get, SSAEnv.set]; dsimp_itree)
  rfl

theorem RHS.sem (X Y: FinInt 32):
    (run (denoteBB _ RHS []) (INPUT X Y) |>.snd.get "r" .i32) =
      (Y - X: FinInt 32) := by
  simp [INPUT, RHS, run, denoteBB, denoteBBStmts]
  rw [ops.subi.sem]
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle]; dsimp_itree
  repeat (simp [SSAEnv.get, SSAEnv.set]; dsimp_itree)

theorem equivalent (X Y: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT X Y) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT X Y) |>.snd.get "r" .i32) := by
  rw [LHS.sem, RHS.sem]; simp
  apply FinInt.comp_sub_comp
end th7

/- LLVM InstCombine: `(add (xor A, B) (and A, B)) --> (or A, B)`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L1411 -/

theorem FinInt.addfull_xor_and (A B: FinInt sz):
    addfull (A ^^^ B) (A &&& B) = .next false (A ||| B) := by
  induction sz with
  | zero => cases A; cases B; decide
  | succ sz ih =>
    match A, B with
    | next bA A', next bB B' =>
      simp [HXor.hXor, HAnd.hAnd, HOr.hOr, xor, and, or, logic2] at *
      simp [addfull, ih]
      cases bA <;> cases bB <;> decide

theorem FinInt.add_xor_and (A B: FinInt sz):
    (A ^^^ B) + (A &&& B) = (A ||| B) := by
  simp [HAdd.hAdd, add, addfull_xor_and]

namespace th8
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "xori"(%A, %B): (i32, i32) -> i32
    %_2 = "andi"(%A, %B): (i32, i32) -> i32
    %r = "addi"(%_1, %_2): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %r = "ori"(%A, %B): (i32, i32) -> i32
]
def INPUT (A B: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("A", ⟨.i32, A⟩), ("B", ⟨.i32, B⟩)
]

theorem equivalent (A B: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT A B) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT A B) |>.snd.get "r" .i32) := by
  simp [LHS, RHS, INPUT]
  simp [run, denoteBB, denoteBBStmt, denoteBBStmts, denoteOp]; dsimp_itree
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; dsimp_itree
  repeat (simp [SSAEnv.get]; dsimp_itree)
  apply FinInt.add_xor_and
end th8

/- LLVM InstCombine: `zext(bool) + C --> bool ? C + 1 : C`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L873 -/

theorem FinInt.add_bool_eq_select (B: FinInt 1) (C: FinInt 32):
    zext 32 B + C = select B (C + 1) C := by
  apply eq_of_toUint_cong2
  simp [cong2, add_toUint]
  rw [zext_toUint' (by decide)]
  cases bool_cases B <;> subst B <;> simp [select, Int.add_assoc, Int.zero_add]
  rw [add_toUint]
  simp [toUint, (by decide: 2^0 = 1), Int.add_zero]
  simp [Int.add_comm]

namespace th9
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "zext"(%B): (i1) -> i32
    %r = "addi"(%_1, %C): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "constant"() {value = 1: i32}: () -> i32
    %_2 = "addi"(%C, %_1): (i32, i32) -> i32
    %r = "select"(%B, %_2, %C): (i1, i32, i32) -> i32
]
def INPUT (B: FinInt 1) (C: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("B", ⟨.i1, B⟩), ("C", ⟨.i32, C⟩)
]

theorem equivalent (B: FinInt 1) (C: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT B C) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT B C) |>.snd.get "r" .i32) := by
  simp [LHS, RHS, INPUT]
  simp [run, denoteBB, denoteBBStmt, denoteBBStmts, denoteOp]; dsimp_itree
  simp [Semantics.semantics_op, arith_semantics_op, AttrDict.find, List.find?,
        AttrEntry.key, AttrEntry.value]; dsimp_itree
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; dsimp_itree
  repeat (simp [SSAEnv.get]; dsimp_itree)
  apply FinInt.add_bool_eq_select
end th9

/- LLVM InstCombine: `(A & ~B) & ~C --> A & ~(B | C)`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAndOrXor.cpp#L1340 -/

theorem FinInt.and_not_and_not (A B C: FinInt sz):
    (A &&& (B ^^^ -1)) &&& (C ^^^ -1) = A &&& ((B ||| C) ^^^ -1) := by
  simp [←comp_eq_xor_minusOne]
  induction sz with
  | zero => cases A; cases B; cases C; decide
  | succ sz ih =>
      match A, B, C with
      | .next bA A', .next bB B', .next bC C' =>
          simp [HAnd.hAnd, HOr.hOr, and, or, comp] at *
          simp [logic2, ih]
          cases bA <;> cases bB <;> cases bC <;> decide

namespace th10
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "constant"() {value = 1: i32}: () -> i32
    %_2 = "negi"(%_1): (i32) -> i32
    %_3 = "xori"(%B, %_2): (i32, i32) -> i32
    %_4 = "andi"(%A, %_3): (i32, i32) -> i32
    %_5 = "xori"(%C, %_2): (i32, i32) -> i32
    %r = "andi"(%_4, %_5): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "constant"() {value = 1: i32}: () -> i32
    %_2 = "negi"(%_1): (i32) -> i32
    %_3 = "ori"(%B, %C): (i32, i32) -> i32
    %_4 = "xori"(%_3, %_2): (i32, i32) -> i32
    %r = "andi"(%A, %_4): (i32, i32) -> i32
]
def INPUT (A B C: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("A", ⟨.i32, A⟩), ("B", ⟨.i32, B⟩), ("C", ⟨.i32, C⟩)
]

theorem LHS.sem (A B C: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT A B C) |>.snd.get "r" .i32) =
      ((A &&& (B ^^^ -1)) &&& (C ^^^ -1): FinInt 32) := by
  simp [INPUT, LHS, run, denoteBB, denoteBBStmts]
  rw [ops.constant.sem]
  rw [ops.negi.sem]
  rw [ops.xori.sem]
  rw [ops.andi.sem]
  rw [ops.xori.sem]
  rw [ops.andi.sem]
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle]; dsimp_itree
  repeat (simp [SSAEnv.get, SSAEnv.set]; simp_itree)
  rfl

theorem RHS.sem (A B C: FinInt 32):
    (run (denoteBB _ RHS []) (INPUT A B C) |>.snd.get "r" .i32) =
      (A &&& ((B ||| C) ^^^ -1): FinInt 32) := by
  simp [INPUT, RHS, run, denoteBB, denoteBBStmts]
  rw [ops.constant.sem]
  rw [ops.negi.sem]
  rw [ops.ori.sem]
  rw [ops.xori.sem]
  rw [ops.andi.sem]
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle]; dsimp_itree
  repeat (simp [SSAEnv.get, SSAEnv.set]; simp_itree)
  rfl

theorem equivalent (A B C: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT A B C) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT A B C) |>.snd.get "r" .i32) := by
  rw [LHS.sem, RHS.sem]; simp
  apply FinInt.and_not_and_not
end th10

/- LLVM InstCombine: `(A & B) | ~(A | B) --> ~(A ^ B)`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAndOrXor.cpp#L1510 -/

theorem FinInt.and_or_not_or (A B: FinInt sz):
    (A &&& B) ||| ((A ||| B) ^^^ -1) = ((A ^^^ B) ^^^ -1) := by
  simp [←comp_eq_xor_minusOne]
  induction sz with
  | zero => cases A; cases B; decide
  | succ sz ih =>
      match A, B with
      | .next bA A', .next bB B' =>
          simp [HXor.hXor, HAnd.hAnd, HOr.hOr, xor, and, or, comp] at *
          simp [logic1, logic2, ih]
          cases bA <;> cases bB <;> decide

namespace th11
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "constant"() {value = 1: i32}: () -> i32
    %_2 = "negi"(%_1): (i32) -> i32
    %_3 = "ori"(%A, %B): (i32, i32) -> i32
    %_4 = "xori"(%_3, %_2): (i32, i32) -> i32
    %_5 = "andi"(%A, %B): (i32, i32) -> i32
    %r = "ori"(%_5, %_4): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "constant"() {value = 1: i32}: () -> i32
    %_2 = "negi"(%_1): (i32) -> i32
    %_3 = "xori"(%A, %B): (i32, i32) -> i32
    %r = "xori"(%_3, %_2): (i32, i32) -> i32
]
def INPUT (A B: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("A", ⟨.i32, A⟩), ("B", ⟨.i32, B⟩)
]

theorem LHS.sem (A B: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT A B) |>.snd.get "r" .i32) =
      ((A &&& B) ||| ((A ||| B) ^^^ -1): FinInt 32) := by
  simp [INPUT, LHS, run, denoteBB, denoteBBStmts]
  rw [ops.constant.sem]
  rw [ops.negi.sem]
  rw [ops.ori.sem]
  rw [ops.xori.sem]
  rw [ops.andi.sem]
  rw [ops.ori.sem]
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle]; dsimp_itree
  repeat (simp [SSAEnv.get, SSAEnv.set]; simp_itree)
  rfl

theorem RHS.sem (A B: FinInt 32):
    (run (denoteBB _ RHS []) (INPUT A B) |>.snd.get "r" .i32) =
      ((A ^^^ B) ^^^ -1: FinInt 32) := by
  simp [INPUT, RHS, run, denoteBB, denoteBBStmts]
  rw [ops.constant.sem]
  rw [ops.negi.sem]
  rw [ops.xori.sem]
  rw [ops.xori.sem]
  simp [interp_ub]; dsimp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle]; dsimp_itree
  repeat (simp [SSAEnv.get, SSAEnv.set]; simp_itree)
  rfl

theorem equivalent (A B: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT A B) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT A B) |>.snd.get "r" .i32) := by
  rw [LHS.sem, RHS.sem]; simp
  apply FinInt.and_or_not_or
end th11

/- LLVM InstCombine: `(X ^ C1) & C2 --> (X & C2) ^ (C1&C2)`
   https://github.com/llvm/llvm-project/blob/291e3a85658e264a2918298e804972bd68681af8/llvm/lib/Transforms/InstCombine/InstCombineAndOrXor.cpp#L1778 -/

theorem FinInt.xor_and (X C₁ C₂: FinInt sz):
    (X ^^^ C₁) &&& C₂ = (X &&& C₂) ^^^ (C₁ &&& C₂) := by
  induction sz with
  | zero => cases X; cases C₁; cases C₂; decide
  | succ sz ih =>
      match X, C₁, C₂ with
      | .next bX X', .next bC₁ C₁', .next bC₂ C₂' =>
          simp [HXor.hXor, HAnd.hAnd, HOr.hOr, xor, and, or] at *
          simp [logic2, ih]
          cases bX <;> cases bC₁ <;> cases bC₂ <;> decide

namespace th12
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "xori"(%X, %C1): (i32, i32) -> i32
    %r = "andi"(%_1, %C2): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "andi"(%X, %C2): (i32, i32) -> i32
    %_2 = "andi"(%C1, %C2): (i32, i32) -> i32
    %r = "xori"(%_1, %_2): (i32, i32) -> i32
]
def INPUT (X C₁ C₂: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("X", ⟨.i32, X⟩), ("C1", ⟨.i32, C₁⟩), ("C2", ⟨.i32, C₂⟩)
]

theorem equivalent (X C₁ C₂: FinInt 32):
    (run (denoteBB _ LHS []) (INPUT X C₁ C₂) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS []) (INPUT X C₁ C₂) |>.snd.get "r" .i32) := by
  simp [LHS, RHS, INPUT]
  simp [run, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp]; simp_itree
  simp [interp_ub]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree
  repeat (simp [SSAEnv.get]; simp_itree)
  apply FinInt.xor_and
end th12
