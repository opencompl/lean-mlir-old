/-
Define the semantics of core Linalg operations.
-/
import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
open MLIR.AST


/-
Consider the following MWE:

```lean
structure DepProof where
   val: Nat
   H: val = 0

def MonadicDepProof [Mon: Monad M]: M DepProof := do
   let v ← pure 0
   return {
      val := v
      H := by {
         /-
         M: Type → Type ?u.380
         Mon: Monad M
         v: ℕ
         ⊢ v = 0
         -/
         sorry
      }
   }
```

How to see in the proof mode that `v` originated from `pure 0`?
-/

instance linalg: Dialect Void Void (fun x => Unit) where
  name := "linalg"
  iα := inferInstance
  iε := inferInstance



def OpM.findIndex (d: AttrDict δ) (key: String): OpM Δ Nat :=
 match d.find_int key with
 | .some ⟨v, MLIRType.index⟩ => return v.toNat
 | _ => OpM.Error s!"{d}.lookup {key} failed to find int"

def OpM.findI32 (d: AttrDict δ) (key: String): OpM Δ (FinInt 32) :=
 match d.find_int key with
 | .some (v, _) => return (FinInt.ofInt 32 v)
 | _ => OpM.Error s!"{d}.lookup {key} failed to find int"


-- in general, xtract slice has offset, size, stride.
-- We ignore the stride and offset for now, just use size.
def linalg_semantics_op: IOp Δ → OpM Δ (TypedArgs Δ)
 | IOp.mk "linalg.extractslice1d" _ [⟨.tensor1d, t⟩] [] dict => do
    let len ←  OpM.findIndex dict "len"
    let t' := t.extract len
    return [⟨.tensor1d, t'⟩]
 | IOp.mk "linalg.fill1d" _ [⟨.tensor1d, t⟩]  [] dict => do
     let cst ←  OpM.findI32 dict "cst"
     let t' := t.fill cst
     return [⟨.tensor1d, t'⟩]

 | IOp.mk "linalg.generic1d" _ [⟨.tensor1d, t⟩] [r] _ => do
      let t' <- t.mapMWithFlatIndex (fun idx val => do
            let rets ← r [⟨.index, idx.ix⟩, ⟨.i32, val⟩]
            match rets with
            | [⟨.i32, v⟩] => pure v
            | _ => OpM.Error s!"linalg.generic1d: unknown return value '{rets}'")
      return [⟨.tensor1d, t'⟩]
 | IOp.mk "linalg.transpose2d"   _ [⟨.tensor1d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.tile2d" _ [⟨.tensor1d, t⟩]  [r] dict => sorry
 | IOp.mk name .. => OpM.Unhandled s!"unhandled {name}"


instance : Semantics linalg where
   semantics_op := linalg_semantics_op
/-
For each transformation, we implement
1) a theorem that proves correctness
2) a test in Test/SemanticTests.lean which tests
   both versions of the program.
-/
namespace ExtractSliceFillCommuteOneD

theorem extract_fill_commute:
 Tensor1D.fill (Tensor1D.extract t extractlen) fillval =
 Tensor1D.extract (Tensor1D.fill t fillval) extractlen := by {
   simp [Tensor1D.fill, Tensor1D.extract];
   apply List.extF
   intros n h; simp; simp at h
   repeat rw [List.getF_replicate]
   . apply Nat.lt_min_left; apply h
   . simp
   . assumption
 }
-- https://mlir.llvm.org/doxygen/BubbleUpExtractSlice_8cpp_source.html
def LHS : Region linalg  := [mlir_region| {
   %x = "linalg.extractslice1d" (%t) { len = 10 : index }: (tensor1d) -> (tensor1d)
   %out = "linalg.fill1d" (%x) { cst = 42 : index }: (tensor1d) -> (tensor1d)
}]
def RHS : Region linalg := [mlir_region| {
   %x = "linalg.fill1d" (%t) { cst = 42 : index }: (tensor1d) -> (tensor1d)
   %out = "linalg.extractslice1d" (%x) { len = 10 : index }: (tensor1d) -> (tensor1d)
}]
/-
TODO: Create a predicate to say that the programs agree on output value `out`.
-/
/-
theorem equiv (t: Tensor1D):
   run ⟦LHS⟧ (SSAEnv.One [ ("t", ⟨.tensor1d, t⟩) ]) =
    run ⟦RHS⟧ (SSAEnv.One [ ("t", ⟨.tensor1d, t⟩) ]) := by {
      simp[LHS, RHS];
      simp_all[denoteRegion, run, StateT.run, denoteTypedArgs, pure, StateT.pure, Except.pure,
            StateT.run, Except.ok, bind, Except.bind, denoteOps, denoteOps
            , StateT.bind, denoteOp, List.mapM, List.mapM.loop, TopM.get,
            StateT.get, OpM.toTopM, TopM.raiseUB, liftM, TopM.set,
            StateT.set, cast];

 }
-/

end ExtractSliceFillCommuteOneD



namespace ExtractSliceGenericCommute1D
variable (r : Region linalg)

-- https://mlir.llvm.org/doxygen/BubbleUpExtractSlice_8cpp_source.html
def LHS: Region linalg  := [mlir_region| {
   %x = "linalg.generic1d" (%t) ($(r)) { len = 10 : index }: (tensor1d) -> (tensor1d)
   %out = "linalg.fill1d" (%x) { cst = 42 : index }: (tensor1d) -> (tensor1d)
}]
def RHS : Region linalg := [mlir_region| {
   %x = "linalg.generic1d" (%t) ($(r)) { cst = 42 : index }: (tensor1d) -> (tensor1d)
   %out = "linalg.extractslice1d" (%x) { len = 10 : index }: (tensor1d) -> (tensor1d)
}]

theorem equiv (t: Tensor1D):
   run ⟦LHS r⟧ (SSAEnv.One [ ("t", ⟨.tensor1d, t⟩) ]) =
    run ⟦RHS r⟧ (SSAEnv.One [ ("t", ⟨.tensor1d, t⟩) ]) := by {
      simp[LHS, RHS];
      simp_all[denoteRegion, run, StateT.run, denoteTypedArgs, pure, StateT.pure, Except.pure,
            StateT.run, Except.ok, bind, Except.bind, denoteOps, denoteOps
            , StateT.bind, denoteOp, List.mapM, List.mapM.loop, TopM.get,
            StateT.get, OpM.toTopM, TopM.raiseUB, liftM, TopM.set,
            StateT.set, cast, mapDenoteRegion, OpM.toTopM, denoteRegion];
      save;
      simp [Semantics.semantics_op];
      sorry
      sorry
      -- simp[linalg_semantics_op];

    }

end ExtractSliceGenericCommute1D

namespace Generic1DFusion

variable (r s : Region linalg)

def LHS: Region linalg  := [mlir_region| {
   %x = "linalg.generic1d" (%t) ($(r)): (tensor1d) -> (tensor1d)
   %y = "linalg.generic1d" (%x) ($(s)): (tensor1d) -> (tensor1d)
}]
def RHS : Region linalg := [mlir_region| {
   %y = "linalg.generic1d" (%t) ({
   ^entry(%x: i32):
     %v1 = "region.run" (%x)  ($(r)) {} : (index) -> (index)
     %v2 = "region.run" (%v1)  ($(s)) {} : (index) -> (index)
     "scf.yield"(%v2): (i32) -> (i32)
   }): (tensor1d) -> (tensor1d)
}]
end Generic1DFusion

namespace Generic1DTiling
variable (r: Region linalg)
-- Need a precondition that the width is divisible by 4.

def LHS: Region linalg  := [mlir_region| {
   %y = "linalg.generic1d" (%x) ($(r)): (tensor1d) -> (tensor1d)
}]
def RHS : Region linalg := [mlir_region| {
   %width = "linalg.dim" (%x)  { "index" = 0 : index } : (tensor1d) -> (index)
   %four = "arith.constant" () { "value" = 4 : index } : () -> (index)
   %num_tiles = "arith.div"(%width , %four) : (index, index) -> (index)
   %y = "scf.for_iter" (%zero, %num_tiles, %x) ({ -- begin, end, loop variable.
     ^entry(%i: index):
       %xchunk = "linalg.extractindex"(%x, %i_times_four, %four) : (tensor1d, index, index) -> (tensor1d)
       %ychunk = "linalg.generic1d" (%xchunk) ($(r)): (tensor1d) -> (tensor1d)
       %yout = "linalg.insertindex"(%x, %i_times_four, %ychunk) : (tensor1d, index, tensor1d) -> (tensor1d)
       "scf.yield"(%yout) : (tensor1d) -> (tensor1d)
   }): (tensor1d) -> (tensor1d)
}]
end Generic1DTiling

namespace Generic2DTiling

end Generic2DTiling

namespace Transpose2D

def LHS: Region linalg  := [mlir_region| {
   %x = "linalg.transpsose2d" (%t) : (tensor2d) -> (tensor2d)
   %y = "linalg.transpose2d" (%x) : (tensor2d) -> (tensor2d)
}]
def RHS : Region linalg := [mlir_region| {
   %y = "scf.id"(%x) : (tensor2d) -> (tensor2d)
}]

end Transpose2D