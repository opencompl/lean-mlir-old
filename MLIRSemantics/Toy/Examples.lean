import MLIR.EDSL
open MLIR.AST

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

-- One of these ways is to simply compute the data when needed, and use the
-- validity proof to back the typing. The type of the operation is not enriched
-- but that also avoids clutter for rare properties.
--
-- In this scheme, the verifier is a function returning the relevant
-- information as an option, returning none if the verification fails. Think
-- for instance a (Vector 2) of arguments, which carries the information that
-- there are exactly 2 arguments, or a list of (SSAVal × MLIRTy), which
-- indicates that the operation type matches the number of arguments.
--
-- The proof of verification says that the verifier returns some value, which
-- is sufficient to later call the function and prove away the case where none
-- is returned. When reading code from disk, the proof of verification can be
-- obtained simply by running the verifier.

def Verifier (α: Type): Type := Op → Option α

def Verifier.ok {α} (v: Verifier α) (o: Op) := Option.isSome (v o)

def Verifier.get {α} {v: Verifier α} {o: Op}: Verifier.ok v o → α :=
  λ (H: Option.isSome (v o)) =>
    match v o, H with
    | some val, _ => val
    | none, H => nomatch H

instance {α} (o: Option α): Decidable (Option.isSome o) :=
  match o with
  | some value => isTrue rfl
  | none => isFalse (λ H => by simp [Option.isSome] at H)

/- === Consistency between arity by argument count and operation type ===

   Assuming custom syntax is expanded, the number of arguments and the
   functional type of the operation must match. Equality can be structurally
   enforced by weaving the lists together, morally changing

     "op"(%arg, %arg, %arg): (!type, !type, !type) -> !ret

   into the woven form

     "op"(%arg:!type, %arg:!type, %arg!type): !ret

   The number of arguments can also be specified at the same time. -/

def zip_args_types (args: List SSAVal) (ty: MLIRTy) :=
  match args, ty with
  | [], MLIRTy.tuple [] =>
      some []
  | a::args, MLIRTy.tuple (t::tys) =>
      Option.map ((a,t) :: .) (zip_args_types args (MLIRTy.tuple tys))
  | [a], t =>
      some [(a,t)]
  | _, _ =>
      none

-- This is the verifier function
def vArgTypeArity: Verifier (List (SSAVal × MLIRTy)) :=
  λ op =>
    match op with
    | Op.mk _ args _ _ _ (MLIRTy.fn ty _) => zip_args_types args ty
    | _ => none

-- Here is an example of how a function can use the proof of validity to ignore
-- operations with non-matching argument/type arity:
def arg_count (o: Op) (H: Verifier.ok vArgTypeArity o): Nat :=
  let args := Verifier.get H;
  args.length

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
      then (arg_count op H) :: count_all_args ops
      else count_all_args ops

-- #eval (count_all_args generic_ops)
-- [1, 2, 0]

/- === Number of SSA values, basic blocks, and regions as arguments ===

   TODO: This time we actually enrich the operation type since it saves the
   need to query; you can just do

     match op with
     | Op.mk _ [arg1,arg2,arg3] _ _ _ _ => ...
     -- no other subcase

  -/


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
