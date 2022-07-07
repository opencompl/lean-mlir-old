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

@[inline, simp, reducible]
def Matrix n m τ :=
  RankedTensor [MLIR.AST.Dimension.Known n, MLIR.AST.Dimension.Known m] τ


-- def Matrix.mk (n m: Nat) (t: Tensor)


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

/-
def makeUniformMLIRTypedArguments [δ: Dialect α σ ε]
  (τ: MLIRType δ):
  List (MLIRType.eval τ) → TypedArgs δ
| [] => []
| t::ts => ⟨τ, t⟩ :: makeUniformMLIRTypedArguments τ ts
-/


-- TODO @lephe: could you perform the mutual induction please?
def MLIRTy_eval_equal_after_coe [Δ: Dialect α σ ε] (τ: MLIRTy):
    τ.eval = (coeMLIRType (c := CoeDialectEmpty (δ := Δ)) τ).eval := sorry

-- TODO @lephe: another proof obligation
def MLIRType_builtin_eval_equal_after_coe [Δ: Dialect α σ ε] [coe: CoeDialect builtin Δ] (τ: MLIRType builtin):
    τ.eval = (coeMLIRType (c := coe) τ).eval := sorry


-- TODO: how do I write the semantics for this in a way that
-- I can get access to the `tensor` type?
def linalg_parallel_iter [Δ: Dialect α σ ε]
   (d1 d2: Nat)
   (inTensor:  Matrix d1 d2 τ)
   (ix1 ix2: Nat): Fitree ((RegionE Δ) +' UBE +' LinalgE) (Option τ.eval) := do
  -- | lol, have fun reasoning with this...
  let data? := inTensor.data.get? (ix1*d2 + ix2)
  match data? with
  | .some data => do
        let out <- Fitree.trigger (RegionE.RunRegion (Δ := Δ) (ix := 0)
                -- TODO, @lephe: please check that my theorem is correct!
                   (args := [⟨ τ, MLIRTy_eval_equal_after_coe τ ▸ data ⟩]))
        match out with
        | [⟨ σ, v ⟩] =>  return (if H: σ = τ then  .some (MLIRTy_eval_equal_after_coe τ ▸ (H ▸ v)) else .none)
        | _ => return .none
  | .none => do
      Fitree.trigger (UBE.DebugUB "unable to access tensor data")
      return .none


def collectOutputsIntoTensorData [δ: Dialect α σ ε]
  (τ: MLIRTy) (argss: List (TypedArgs δ)): List τ.eval :=
  match argss with
  | [] => []
  | (args::argss) => match args with  -- TODO: fix this semantics
               | [⟨τ', v⟩] => if H: τ = τ' then [] else []
               | _ => []

-- This is haskell's `traverse: (Traversable t, Applicative f) => t (f a) -> f (t a)`
-- specialize for `t = List`, `f = Option`
def list_option_to_option_list (xs: List (Option α)): Option (List α) :=
  match xs with
  | [] => .some []
  | (.some x)::xs =>  (list_option_to_option_list xs).map (fun xs' => x::xs' )
  | .none::xs => .none

#check RankedTensor.mk
def linalg_parallel_all_iters
  [CoeDialect builtin Δ]
   (d1 d2: Nat)
   (inTensor: Matrix d1 d2 τ):
     Fitree ((RegionE Δ) +' UBE +' LinalgE) (TypedArgs Δ) := do
  -- | TODO: Yeesh, we gotta worry about List.bind.
  let ixs : List (Nat × Nat) := (List.range d1).bind (fun ix1 => (List.range d2).map (fun ix2 => (ix1, ix2)))
  let outValues <- ixs.mapM (fun ix2d => linalg_parallel_iter d1 d2 inTensor ix2d.fst ix2d.snd)
  let outValues := list_option_to_option_list outValues
  match outValues with
  | .some outValues =>
        -- | TODO:
        let t : Tensor τ:= { inTensor.toTensor with data := outValues, h_data_size := sorry }
        let dims : DimList :=  [Dimension.Known d1, Dimension.Known  d2]
        let out_tensor_τ := builtin.tensor dims τ
        let out_tensor := RankedTensor.mk (D := dims) (toTensor := t) sorry
        return [⟨out_tensor_τ, MLIRType_builtin_eval_equal_after_coe out_tensor_τ ▸ out_tensor⟩]
  | .none => do
      Fitree.trigger $ UBE.DebugUB "RankedTensor: unable to produce output args."
      return []


-- def toy_semantics_op (ret_name: Option SSAVal) (op: Op builtin):
-- | TODO: we need a way to say that `builtin` is a member of Gδ
def linalg_semantics_op  [CoeDialect builtin Δ] [P: DialectProjection Δ builtin]: IOp Δ →
      Option (Fitree (RegionE Δ +' UBE +' LinalgE) (BlockResult Δ))
  | IOp.mk "linalg.parallel2d1" [⟨.extended sΔ, v⟩] [] 1 _ _ => do
      match H: DialectProjection.project_σ (self := P) _ _ sΔ with
      | some (builtin.σ.tensor [Dimension.Known d1, Dimension.Known d2] τ) => .some do
          let input: RankedTensor [Dimension.Known  d1, Dimension.Known  d2] τ :=
            cast (by rw [H]) <| DialectProjection.project_ε (self := P) sΔ v
          let out  <- linalg_parallel_all_iters d1 d2 input
          return (BlockResult.Ret out)
      | _ => none
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
