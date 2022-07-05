/-
## `linalg` dialect

This file formalises part of the `linalg` dialect. 
The key concepts we model are that of parallel loops with lower
and upper bounds as described by linalg.

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

`linalg` has no extended types or attributes.
-/

instance linalg: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

/-
### Dialect operations
-/

inductive LinalgE: Type → Type :=
| GenericParallel: LinalgE (RankedTensor D τ) 
  

def List.tailList (xs: List α): List α :=
  match xs with 
  | [] => []
  | (x::xs') => xs'

-- snoc is 'cons' at the end / 'cons' on the reverse of the list.
def List.snoc (xs: List α) (a: α): List α := 
  match xs with 
  | [] => [a]
  | (x::xs') => x::(xs'.snoc a)

def zip_same_size (xs: List α) (ys: List β): Option (List (α × β)) :=
 match xs with 
 | [] => match ys with 
         | [] => .some []
         | _ => .none
 | (x::xs') => match ys with 
         | [] => .none
         | (y::ys') => (zip_same_size xs' ys').map (fun zipped => (x,y)::zipped)

def List.sum (xs: List Nat): Nat := 
  xs.foldl (fun x y => x + y) 0

def List.pointwiseMul (xys: List (Nat × Nat)): List Nat := 
  xys.map (fun xy => xy.fst * xy.snd) 


-- xs[α] for xs of shape 10: α
-- xs[α][β][γ] for xs of shape 40x50x60: α*(50*60)+ β*60 + γ*1 ~= [50, 1] *+ [α, β]
-- xs[α][β][γ] for xs of shape 40x50x60: ((0 + α)*50 + β)*60 + γ)*1 ~= [50, 1] *+ [α, β]
def linearizeIndex (shape: List Nat) (ix: List Nat): Option Nat := 
  (zip_same_size ix (shape.drop 1)).map $ List.foldr (init := 0)
    (fun ix_and_shape linix => (linix + ix_and_shape.fst) * ix_and_shape.snd)

def makeUniformMLIRTypedArguments [δ: Dialect α σ ε]
  (τ: MLIRType δ):
  List (MLIRType.eval τ) → TypedArgs δ
| [] => []
| t::ts => ⟨τ, t⟩ :: makeUniformMLIRTypedArguments τ ts



#check MLIRType.eval
-- TODO: how do I write the semantics for this in a way that 
-- I can get access to the `tensor` type?
def linalg_parallel_iter [Δ: Dialect α σ ε]
   (inTensors: List (Tensor τ))
   (ix: Nat):
     Fitree ((RegionE Δ) +' UBE +' LinalgE)
            (TypedArgs Δ) := do
  let data? := inTensors.mapM (fun inTensor => inTensor.data.get? ix)
  match data? with 
  | .some data => do
      Fitree.trigger (RegionE.RunRegion (Δ := Δ) (ix := 0)
       (args := makeUniformMLIRTypedArguments 
                  (δ := Δ)
                  (coeDialectType.coe τ)
                  (coe_type_eval_eq τ ▸ data)))
  | .none => do 
      Fitree.trigger (UBE.DebugUB "unable to access tensor data")
      return []


def collectOutputsIntoTensorData [δ: Dialect α σ ε]
  (τ: MLIRTy) (argss: List (TypedArgs δ)): List τ.eval :=
  match argss with 
  | [] => []
  | (args::argss) => match args with  -- TODO: fix this semantics
               | [⟨τ', v⟩] => if H: τ = τ' then [] else []
               | _ => []

def collectOutputsIntoTensor [δ: Dialect α σ ε]
  (shape: Nat)
  (τ: MLIRTy) (argss: List (TypedArgs δ)): Tensor τ :=
  Tensor.mk [shape]  (collectOutputsIntoTensorData τ argss) sorry

def linalg_parallel_all_iters
  [CoeDialect builtin Δ]
    (inTensors: List (Tensor τ))
   (size: Nat): 
     Fitree ((RegionE Δ) +' UBE +' LinalgE)
            (TypedArgs Δ) := do
  let ixs := List.range size
  let outValues <- ixs.mapM (linalg_parallel_iter inTensors)
  return [⟨builtin.tensor_unranked τ,
           (coe_type_eval_eq (builtin.tensor_unranked τ)) ▸ 
            collectOutputsIntoTensor size τ outValues⟩]


-- def toy_semantics_op (ret_name: Option SSAVal) (op: Op builtin):
-- | TODO: we need a way to say that `builtin` is a member of Gδ
def linalg_semantics_op [CoeDialect builtin Δ]: IOp Δ →
      Option (Fitree (RegionE Δ +' UBE +' LinalgE) (BlockResult Δ))
  | IOp.mk "linalg.parallel1d1" [⟨builtin.tensor_unranked τ, xs⟩] [] 1 _ _ => some do
          linalg_parallel_all_iters [input]
  | IOp.mk "linalg.parallel1d2" [input1, input2] [] 1 _ _ => some do
      sorry

  | _ => none

/-
Hook to provide a custom AffineMap used to construct the
hyperrectangular loop iteration space given all the operand subshapes.
This is used to answer the question:
"Given a list of operand ranges, what is the subportion of the iteration
space involved in the computation".
This is the inverse problem of `getLoopsToShapesMap`.
Return the empty AffineMap when such an AffineMap cannot be constructed.
The default behavior is based on a very simple inference procedure that
only works with permutation affine maps.
A more advanced Tensor-Comprehension like inference is possible but has
proven to be ambiguous in unfavorable case.
A safer and more robust alternative is to allow each op to define
its own AffineMap.
-/

#check RankedTensor
def LinalgE.handle [δ: Dialect α σ ε] {E}: LinalgE ~> Fitree E := fun T e =>
   match e with 
    | .GenericParallel => sorry
/-
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

#eval run ⟦cst1⟧ (SSAEnv.empty (δ := arith))


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
  simp [run, semantics_bbstmt, semantics_op!, Semantics.semantics_op]
  simp [arith_semantics_op, Semantics.handle, add1, add2]
  simp [interp_ub!]; simp_itree
  simp [interp_ssa]; simp_itree
  rw [FinInt.add_comm]
-/
