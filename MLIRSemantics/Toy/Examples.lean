import MLIR.EDSL
import MLIRSemantics.Toy.Toy
import MLIRSemantics.Verifier
import MLIRSemantics.SSAEnv
open MLIR.AST

-- SSAEnv tests

private def example_env (l: SSAScope) (s: SSAEnv): SSAEnv :=
  (⟨"%0", MLIRTy.int 32, 42⟩ :: ⟨"%1", MLIRTy.int 32, 7⟩ :: l) ::
  (⟨"%3", MLIRTy.int 32, 12⟩ :: ⟨"%r", MLIRTy.float 32, -1.7e3⟩ :: []) ::
  s

example: ∀ l s, SSAEnv.get "%1" (MLIRTy.int 32) (example_env l s) = some 7 :=
  by simp

-- Verifier tests

def funcDoubleTranspose: Op := [mlir_op|
  "func"() ({
    ^bb0:
      %t0 = "toy.const"() {value = dense<[1,2,3,4]>: i32}: () -> tensor<1×4:i32>
      %t1 = "toy.transpose"(%t0): tensor<1×4:i32> -> tensor<4×1:i32>
      %t2 = "toy.transpose"(%t1): tensor<4×1:i32> -> tensor<1×4:i32>
     "std.return"(%t2): tensor<1×4:i32> -> ()
  }): () -> ()
]
-- #reduce funcDoubleTranspose

/- The following are WIP/tests about enriching the operand type so that the
   semantics can be defined *under the assumption that the verifier validates
   the IR*, which is fairly important to avoid the giant overhead of the
   generic operation/bb/region format.

   There are a number of verifications/constraints that we'd like to enforce
   through typing and/or hypotheses, in whichever way is the most convenient
   for the dialect developer. -/

-- And how arbitrary code can be checked with the decidable if.
def opArity_valid1: Op := [mlir_op|
  "toy.operation"(%t1): i32 -> i32
]
def opArity_valid2: Op := [mlir_op|
  "toy.operation"(%t1,%t2): (f32,f32) -> i32
]
def opArity_valid3: Op := [mlir_op|
  "toy.operation"(): () -> i32
]
def opArity_mismatched1: Op := [mlir_op|
  "toy.operation"(%t1,%t2): () -> ()
]
def opArity_mismatched2: Op := [mlir_op|
  "toy.operation"(%i0): (i32,i32) -> ()
]
def generic_ops :=
  [opArity_valid1, opArity_valid2, opArity_valid3, opArity_mismatched1,
   opArity_mismatched2]
def count_all_args ops :=
  match ops with
  | [] => []
  | op::ops =>
      if H: Verifier.ok vArgTypeArity op
      then (Verifier.get H).length :: count_all_args ops
      else count_all_args ops

-- #eval (count_all_args generic_ops)
-- [1, 2, 0]

-- Abusing dependent typing on the generic format

/- def argtype: (x: String) → Type
  | "toy.whatever" => Nat
  | "toy.constant" => (Nat × Nat)
  | "toy.transpose" => (n: Nat) × (m: Nat) × Tensor Nat [n,m]
  | _ => Unit

def rettype: (x: String) → argtype x → Type
  | "toy.whatever", _ => Nat
  | "toy.constant", _ => Nat
  | "toy.transpose", ⟨n, m, _⟩ => Tensor Nat [m,n]
  | _, _ => Unit

def semantics: (x: String) → (a: argtype x) → rettype x a
  | "toy.whatever", n => n
  | "toy.constant", (p,q) => p + q
  | "toy.transpose", ⟨n, m, t⟩ =>
      -- Here t has type [Tensor Nat [n,m]] (not even something definitionally
      -- equal, really exactly that)
      transpose t
  | _, _ => /- fails here -/ () -/

/- === Other ideas for constraints === -/

def opConst_valid: Op := [mlir_op|
  "toy.const"() {value = dense<[1,2,3,4]>: i32}: () -> tensor<1×4:i32>
]
def opConst_badValue: Op := [mlir_op|
  "toy.const"() {value = dense<[[1],[2],[3,4],[5]]>: i32}: () -> tensor<1×4:i32>
]
def opConst_badShape: Op := [mlir_op|
  "toy.const"() {value = dense<[1,2,3]>: i32}: () -> tensor<3×5:i32>
]
def opConst_noValue: Op := [mlir_op|
  "toy.const"(): () -> tensor<1×2:f32>
]
def opConst_hasArgs: Op := [mlir_op|
  "toy.const"(%t1) {value = dense<[[1],[3],[5]]>: i32}:
    tensor<1×3:i32> -> tensor<3×1:i32>
]
