import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
open MLIR.AST

/-
### Dialect: `scf`
-/

instance scf: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

-- Operations of `scf` that unfold into regions need to expose these regions
-- immediately at denotation time rather than at interpretation time, so that
-- inner operations can be interpreted correctly. So operations with regions
-- are not represented here, and handled directly in `semantics_op`.
inductive ScfE: Type -> Type :=

-- | run a loop, decrementing i from n to -
-- | ix := lo + (n - i) * step
def run_loop_bounded_go [Monad m] (n: Nat) (i: Nat) (lo: Int) (step: Int)
  (accum: a) (eff: Int -> a -> m a): m a := do
   let ix : Int := lo + (n - i) * step
   let accum <- eff ix accum
   match i with
   | .zero => return accum
   | .succ i' => run_loop_bounded_go n i' lo step accum eff

-- | TODO: use this to run regions.
def run_loop_bounded [Monad m] (n: Nat) (lo: Int) (step: Int) (accum: a) (eff: Int -> a -> m a): m a :=
  run_loop_bounded_go n n lo step accum eff

-- | TODO: refactor to (1) an effect, (2) an interpretation
-- | TODO: use the return type of Scf.For. For now, just do unit.
def scf_semantics_op: IOp Δ →
      Option (Fitree (RegionE Δ +' UBE +' ScfE) (BlockResult Δ))
  | IOp.mk "scf.if" [⟨.i1, b⟩] [] 2 _ _ =>
      some (Fitree.trigger <| RegionE.RunRegion (if b == 1 then 0 else 1) [])
  | IOp.mk "scf.for" [⟨.i32, lo⟩, ⟨.i32, hi⟩, ⟨.i32, step⟩] [] 1 _ _ => some do
    let nsteps : Int := ((FinInt.toSint'  hi) - (FinInt.toSint' lo)) / FinInt.toSint' step
    run_loop_bounded
      (a := BlockResult Δ)
      (n := nsteps.toNat)
      (lo := (FinInt.toSint' lo))
      (step := (FinInt.toSint' step))
      (accum := default)
      (eff := (fun i _ => Fitree.trigger <| RegionE.RunRegion 0 []))
  | IOp.mk "scf.yield" vs [] 0 _ _ =>
    some <| return BlockResult.Ret vs
  | _ => none

def handleScf: ScfE ~> Fitree Void1 :=
  fun _ e => nomatch e

instance: Semantics scf where
  E := ScfE
  semantics_op := scf_semantics_op
  handle := handleScf

/-
### Theorems
-/

namespace scf_th1
-- Proof that `scf.if` with a fixed condition simplifies to its "then" or
-- "else" region depending on the value

def LHS (r₁ r₂: Region scf): Region scf := [mlir_region|
{
  "scf.if" (%b) ($(r₁), $(r₂)) : (i1) -> ()
}]
def INPUT (b: Bool): SSAEnv scf := SSAEnv.One [
  ⟨"b", MLIRType.i1, if b then 1 else 0⟩
]

-- Pure unfolding-style proof
theorem equivalent (b: Bool):
    run (denoteRegion _ (LHS r₁ r₂) []) (INPUT b) =
    run (denoteRegion _ (if b then r₁ else r₂) []) (INPUT b) := by
  simp [LHS, INPUT, denoteRegion, denoteBB, denoteBBStmts, denoteTypedArgs]
  simp [denoteBBStmt, denoteOp, List.zip, List.zipWith, List.mapM]
  simp [Semantics.semantics_op, scf_semantics_op]
  simp [interpRegion, denoteRegions]
  simp [run, interpUB_bind, interpSSA'_bind]
  conv in interpSSA' (Fitree.trigger _) => simp [Fitree.trigger]
  simp [SSAEnvE.handle, cast_eq]
  cases b <;> simp [List.get!]
end scf_th1
