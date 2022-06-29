import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
open MLIR.AST

/-
### Dialect: `dummy`
-/

instance scf: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

-- | interesting, what is the type of ScfFor?
-- | TODO: use the return type. For now, just do unit.
-- Fitree S [S: Semantics δ]
inductive ScfE  : Type -> Type := 
  | For: (low:Int) → (upper: Int) → (step: Int) → ScfE Unit

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
#check Signedness
def semantics_op:
    IOp scf →
    Fitree (RegionE +' UBE +' (SSAEnvE scf) +' ScfE) (BlockResult scf)
  | IOp.mk "scf.for" [⟨.int .Signless 32, lo⟩, ⟨.int .Signless 32, hi⟩, ⟨.int .Signless 32, step⟩] [] 1 _ _ => do 
    let nsteps : Int := ((FinInt.toSint'  hi) - (FinInt.toSint' lo)) / FinInt.toSint' step
    let out <- run_loop_bounded (a := PUnit)
                 (n := nsteps.toNat)
                 (lo := (FinInt.toSint' lo))
                 (step := (FinInt.toSint' step))
                 (accum := PUnit.unit)
                 (eff := (fun i _ => (Fitree.trigger (RegionE.runRegion 0)))) -- how to type this correctly?
    return BlockResult.Next ⟨.unit, ()⟩
    
/-


def scf_semantics_op (Δ: Dialect α σ ε) [SΔ: Semantics Δ]:
        Op (scf + Δ) → Option (Fitree (SSAEnvE (scf + Δ) +' (ScfE Δ +' SΔ.E +' UBE)) (BlockResult (scf + Δ)))
  | Op.mk "scf.for" [lo, hi, step] _ [r] _ (.fn (.tuple []) (.int sgn sz)) => some do
      let lo : FinInt sz <- SSAEnv.get? (scf + Δ) (MLIRType.int sgn sz) lo
      let hi : FinInt sz <- SSAEnv.get? (scf + Δ) (MLIRType.int sgn sz) hi
      let step : FinInt sz <- SSAEnv.get? (scf + Δ) (MLIRType.int sgn sz) step
      let rsem := semantics_region_single_bb r
      let t <- Fitree.trigger (ScfE.For (Δ := Δ) (FinInt.toSint' lo) (FinInt.toSint' hi) (FinInt.toSint' step) r);
      -- let nsteps : Int := ((FinInt.toSint'  hi) - (FinInt.toSint' lo)) / FinInt.toSint' step
      -- let out <- run_loop_bounded (a := PUnit)
      --            (n := nsteps.toNat)
      --            (lo := (FinInt.toSint' lo))
      --            (step := (FinInt.toSint' step))
      --            (accum := PUnit.unit)
      --            (eff := (fun i _ => (semantics_region_single_bb r))) -- how to type this correctly?
                 -- (eff := (fun i _ => pure PUnit.unit))
      -- let i ← Fitree.trigger (ScfE.For 0 0 0)
      -- SSAEnv.set? (δ := Gδ) (.int sgn sz) ret (.ofInt sgn sz i)
      return BlockResult.Next ⟨.unit, ()⟩
  | _ => none
-/

private def eff_inject {E} [Semantics δ] (x: Fitree (UBE +' SSAEnvE δ +' Semantics.E δ) Unit):
    Fitree (UBE +' SSAEnvE δ +' Semantics.E δ +' E) PUnit :=
  let y: Fitree (UBE +' SSAEnvE δ +' Semantics.E δ +' E) Unit :=
    Fitree.translate (fun t v =>  Member.inject _ v) x
  let z : Fitree (UBE +' SSAEnvE δ +' Semantics.E δ +' E) PUnit :=
    Functor.map (fun _ => PUnit.unit) y
  z


def handleScf {E} [Semantics δ]: ScfE ~> Fitree PVoid
  fun _ e => return ()
  /-
    match e with
    | .For lo hi step r => do
      let nsteps : Int := (hi - lo) / step
      let out <- run_loop_bounded (a := PUnit)
                 (n := nsteps.toNat)
                 (lo := lo)
                 (step := step)
                 (accum := PUnit.unit)
                 (eff := (fun i _ => eff_inject (semantics_region_single_bb (Gδ := δ) r)))
       return ()
  -/
-- set_option pp.all true
instance: Semantics scf where
  E := ScfE

  semantics_op := scf_semantics_op
  handle := handleScf

/-
### Examples and testing
-/


