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


set_option maxHeartbeats 9999999999

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


-- We assume that we only run regions that behave purely (otherwise most of the
-- theorems on generic don't work).
def validGenericRegion {Δ: Dialect α σ ε} (r: Region Δ) (f: Int → FinInt 32 → FinInt 32) :=
  forall (i: Int) (v: FinInt 32),
  OpM.denoteRegion r 0 [⟨.index, i⟩, ⟨.i32, v⟩] = return [⟨.i32, f i v⟩]


def OpM.findIndex (d: AttrDict δ) (key: String): OpM Δ Nat :=
 match d.find_int key with
 | .some ⟨v, MLIRType.index⟩ => return v.toNat
 | _ => OpM.Error s!"{d}.lookup {key} failed to find int"

def OpM.findI32 (d: AttrDict δ) (key: String): OpM Δ (FinInt 32) :=
 match d.find_int key with
 | .some (v, _) => return (FinInt.ofInt 32 v)
 | _ => OpM.Error s!"{d}.lookup {key} failed to find int"

#check decide

-- in general, xtract slice has offset, size, stride.
-- We ignore the stride and offset for now, just use size.
def linalg_semantics_op {Δ: Dialect α σ ε}: IOp Δ → OpM Δ (TypedArgs Δ)
 | IOp.mk "linalg.extractslice1d" _ [⟨.tensor1d, t⟩] [] dict => do
   let len ←  OpM.findIndex dict "len"
   if H: len ≤ t.size0 then
      let t' := t.extract len H
      return [⟨.tensor1d, t'⟩]
   else
      OpM.Error s!"expected {len} ≤ {t.size0}."
 | IOp.mk "linalg.fill1d" _ [⟨.tensor1d, t⟩]  [] dict => do
     let cst ←  OpM.findIndex dict "cst"
     let t' := t.fill cst
     return [⟨.tensor1d, t'⟩]
 | IOp.mk "linalg.generic1d'" _ [⟨.tensor1d, t⟩] [r] _ => do -- generic1d without array index.
      let t' <- t.mapM (fun val => do
            let rets ← r [⟨.index, val⟩]
            match rets with
            | [⟨.index, v⟩] => pure v
            | _ => OpM.Error s!"linalg.generic1d: unknown return value '{rets}'")
      return [⟨.tensor1d, t'⟩]
 | IOp.mk "linalg.generic1d" _ [⟨.tensor1d, t⟩] [r] _ => do
      let t' <- t.mapMWithFlatIndex (fun validx => do
            let val := validx.fst
            let idx := validx.snd
            let rets ← r [⟨.index, idx⟩, ⟨.index, val⟩]
            match rets with
            | [⟨.index, v⟩] => pure v
            | _ => OpM.Error s!"linalg.generic1d: unknown return value '{rets}'")
      return [⟨.tensor1d, t'⟩]
 | IOp.mk "linalg.extractslice2d" _ [⟨.tensor2d, t⟩]  [r] dict =>  do
      let len0 ←  OpM.findIndex dict "len0"
      let len1 ←  OpM.findIndex dict "len1"
      dite (α := OpM Δ (TypedArgs Δ))
         (len0 <= t.size0)
         (fun LEQ0 =>
            dite (len1 <= t.size1)
               (fun LEQ1 => do
                  let subview : TensorSubview2D t.size0 t.size1
                     := { size0 := len0, size1 := len1, IX0 := LEQ0, IX1 := LEQ1}
                  let t' := t.extractslice' subview
                  return [⟨.tensor2d, t'⟩])
               (fun GT0 => OpM.Error "expected index inbounds"))
         (fun GT1 => OpM.Error "expected index inbounds")
 | IOp.mk "linalg.fill2d" _ [⟨.tensor2d, t⟩]  [r] dict => do
     let cst ←  OpM.findI32 dict "cst"
     let t' := t.fill (FinInt.toSint cst)
     return [⟨.tensor2d, t'⟩]
 | IOp.mk "linalg.transpose2d"   _ [⟨.tensor2d, t⟩]  [r] dict => do
     let t' := t.transpose
     return [⟨.tensor2d, t'⟩]
 | IOp.mk "linalg.insertslice2d" _ [⟨.tensor2d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.generic2d" _ [⟨.tensor2d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.parallel2d" _ [⟨.tensor2d, t⟩]  [r] dict => do
      return []
 | IOp.mk name .. => OpM.Unhandled s!"unhandled {name}"

theorem linalg_semantics_generic1d {Δ: Dialect α σ ε} {r: Region Δ} {rSpec types attrs}:
  validGenericRegion r rSpec →
  linalg_semantics_op (Δ := Δ)
      (IOp.mk "linalg.generic1d" types [⟨.tensor1d, t⟩] [OpM.denoteRegion r 0] attrs) =
    return [⟨.tensor1d, t.mapWithFlatIndex (fun idxval => rSpec idxval.fst idxval.snd)⟩] := by
  intros h
  simp [linalg_semantics_op]
  simp [validGenericRegion] at h
  simp [h]
  rw [Tensor1D.mapM_map]
  . rfl
  . intros; rfl

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

-- {pre} TopM {post}
-- {x1 = 10 * x2 = 30 * x4 = 50}

-- https://mlir.llvm.org/doxygen/BubbleUpExtractSlice_8cpp_source.html
def LHS: Region linalg  := [mlir_region| {
   %x = "linalg.generic1d" (%t) ($(r)) { len = 10 : index }: (tensor1d) -> (tensor1d)
   %out = "linalg.fill1d" (%x) { cst = 42 : index }: (tensor1d) -> (tensor1d)
}]
def RHS : Region linalg := [mlir_region| {
   %x = "linalg.generic1d" (%t) ($(r)) { cst = 42 : index }: (tensor1d) -> (tensor1d)
   %out = "linalg.extractslice1d" (%x) { len = 10 : index }: (tensor1d) -> (tensor1d)
}]

@[simp]
theorem OpM.bind_ret {Δ: Dialect α σ ε} (a: α) (k: α → OpM Δ β):
  bind (OpM.Ret a) k = k a := rfl

@[simp]
theorem OpM.toTopM_pure {r: TypedArgs Δ}:
  OpM.toTopM rs (pure r) env = Except.ok (r, env) := rfl

theorem equiv (t: Tensor1D) r rSpec:
   validGenericRegion r rSpec →
   run ⟦LHS r⟧ (SSAEnv.One [ ("t", ⟨.tensor1d, t⟩) ]) =
   run ⟦RHS r⟧ (SSAEnv.One [ ("t", ⟨.tensor1d, t⟩) ]) := by {
      intros valid_r;
      simp[LHS, RHS];
      simp_all[denoteRegion, run, StateT.run, List.map, denoteTypedArgs, pure, StateT.pure, Except.pure,
            StateT.run, Except.ok, bind, Except.bind, denoteOps, denoteOps
            , StateT.bind, denoteOp, List.mapM, List.mapM.loop, TopM.get,
            StateT.get, OpM.toTopM, TopM.raiseUB, liftM, TopM.set,
            StateT.set, cast, OpM.denoteRegions, TopM.mapDenoteRegion,
             OpM.toTopM, denoteRegion] (config := { maxSteps := 99999999 });
      -- save;
      simp [Semantics.semantics_op];
      -- rewrite [linalg_semantics_generic1d];
      sorry
    }
end ExtractSliceGenericCommute1D

namespace mapMCommute
def fish [Monad m] (f: a -> m b) (g: b -> m c): a -> m c := fun a => (f a) >>= g

theorem mapM_cons [M: Monad m] [LM: LawfulMonad m] (x: a) (xs: List a) (f: a -> m b):
  List.mapM f (x :: xs) =  f x >>= (fun b => do let bs <- List.mapM f xs; pure (b :: bs)) := by {
  simp[List.mapM, List.mapM.loop];
  sorry;
}
theorem commute_implies_mapM_commute [Monad m] [LawfulMonad m]
  (f g : a -> m a ) (k: List a -> a -> m b)
  (COMMUTE: forall {b: Type} (x y : a)  (k: a -> a -> m b), f x >>= (fun r1 => (g y >>= fun r2 => k r1 r2 )) =
                                        g y >>= fun r2 => f x >>= fun r1 => k r1 r2):
  List.mapM f xs >>= (fun r1 => (g y >>= fun r2 => k r1 r2 )) =
                                        g y >>= fun r2 => List.mapM f xs >>= fun r1 => k r1 r2 := by sorry

theorem mapM_commute [M: Monad m] [LM: LawfulMonad m]
  (f g: a -> m a) (COMMUTE: forall {b : Type} (x y : a)  (k: a -> a -> m b), f x >>= (fun r1 => (g y >>= fun r2 => k r1 r2 )) =
                                        g y >>= fun r2 => f x >>= fun r1 => k r1 r2)
  : fish (List.mapM f) (List.mapM g) = List.mapM (fish f g) := by {

     funext x;
     induction x;
     case nil => {
          simp[fish, List.mapM, List.mapM.loop];
     }
     case cons head tail IH => {
       simp[fish];
       rewrite [mapM_cons];
       rewrite [mapM_cons];

       simp [bind_assoc]
       simp[mapM_cons];
       congr; -- remove the head
       funext x';
       rewrite [commute_implies_mapM_commute];
       congr;
       funext x';
       simp [fish] at IH;
       rewrite [<- IH];
       simp[bind_assoc];
       apply COMMUTE;

     }

}

end mapMCommute

namespace Generic1DFusion


variable (r s : Region linalg)

-- See section MapMCommute.
-- See section MapMCommute
-- fmap f . fmap g == fmap (f . g)
-- true iff f commutes with g?
-- mapM f >=> mapM g == mapM (f >=> g)
-- naive proof: induction on size of %t.
-- god's proof in the book: show that the computation of y[i] looks like
--     as what's written below. (need some notion of funext on the array).
def LHS: Region linalg  := [mlir_region| {
   %x = "linalg.generic1d" (%t) ($(r)): (tensor1d) -> (tensor1d) -- fmap r | x[i] <- r(t(i))
   %y = "linalg.generic1d" (%x) ($(s)): (tensor1d) -> (tensor1d) -- fmap s | y[i] <- s(x[i]) = s(r(t(i)))
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
