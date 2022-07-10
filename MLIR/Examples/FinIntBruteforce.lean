import MLIR.Util.FinInt
import MLIR.Dialects.ArithSemantics
open MLIR.AST

abbrev FinIntPred1 := (sz: Nat) → FinInt sz → Bool
abbrev FinIntPred2 := (sz: Nat) → FinInt sz → FinInt sz → Bool
abbrev FinIntPred3 := (sz: Nat) → FinInt sz → FinInt sz → FinInt sz → Bool

def isTautologyUpTo1 (sz: Nat) (P: FinIntPred1): Bool :=
  match sz with
  | 0 => P 0 .nil
  | sz+1 =>
      isTautologyUpTo1 sz (fun sz n => P (sz+1) n.O) &&
      isTautologyUpTo1 sz (fun sz n => P (sz+1) n.I)

def isTautologyUpTo2 (sz: Nat) (P: FinIntPred2): Bool :=
  match sz with
  | 0 => P 0 .nil .nil
  | sz+1 =>
      isTautologyUpTo2 sz (fun sz n m => P (sz+1) n.O m.O) &&
      isTautologyUpTo2 sz (fun sz n m => P (sz+1) n.O m.I) &&
      isTautologyUpTo2 sz (fun sz n m => P (sz+1) n.I m.O) &&
      isTautologyUpTo2 sz (fun sz n m => P (sz+1) n.I m.I)

def P₁: FinIntPred2 := fun _ X Y =>
  ((X ||| Y) - X) = ((X ^^^ -1) &&& Y)

theorem P₁_tautology8: isTautologyUpTo2 8 P₁ := by
  native_decide

def P₂: FinIntPred2 := fun _ X Y =>
  (X + Y) - (X &&& Y) = (X ||| Y)

theorem P₂_tautology8: isTautologyUpTo2 8 P₂ := by
  native_decide

def P₃: FinIntPred2 := fun _ X Y =>
  -(FinInt.select 0 (-X) Y) = FinInt.select 0 X (-Y) &&
  -(FinInt.select 1 (-X) Y) = FinInt.select 1 X (-Y)

theorem P₃_tautology8: isTautologyUpTo2 8 P₃ := by
  native_decide

--

axiom alive1 (P: FinIntPred1):
  isTautologyUpTo1 8 P → ∀ sz n, P sz n

axiom alive2 (P: FinIntPred2):
  isTautologyUpTo2 8 P → ∀ sz n m, P sz n m

---

namespace BruteforceThm1
def LHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %_1 = "addi"(%X, %Y): (i32, i32) -> i32
    %_2 = "andi"(%X, %Y): (i32, i32) -> i32
    %r = "subi"(%_1, %_2): (i32, i32) -> i32
]
def RHS: BasicBlock arith := [mlir_bb|
  ^bb:
    %r = "ori"(%X, %Y): (i32, i32) -> i32
]
def INPUT (X Y: FinInt 32): SSAEnv arith := SSAEnv.One [
  ("X", ⟨.i32, X⟩), ("Y", ⟨.i32, Y⟩)
]

-- Too long... times out during type checking
/-
theorem equivalent (X Y: FinInt 32):
    (run (denoteBB _ LHS) (INPUT X Y) |>.snd.get "r" .i32) =
    (run (denoteBB _ RHS) (INPUT X Y) |>.snd.get "r" .i32) := by
  simp [LHS, RHS, INPUT]
  simp [run, denoteBB, denoteBBStmt, denoteOp]; simp_itree
  simp [interp_ub]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree
  repeat (simp [SSAEnv.get]; simp_itree)
  have h := alive2 _ P₂_tautology8 _ X Y
  simp [P₂] at h; assumption
-/
end BruteforceThm1
