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
  | IOp.mk "scf.if" [⟨.i1, b⟩] [] 2 _ _ => some do
      if b == 0
      then Fitree.trigger <| RegionE.RunRegion 0 []
      else Fitree.trigger <| RegionE.RunRegion 1 []
  | IOp.mk "scf.for" [⟨.index, lo⟩, ⟨.index, hi⟩, ⟨.index, step⟩] [] 1 _ _ => some do
    let nsteps : Int := (hi - lo) / step
    run_loop_bounded
      (a := BlockResult Δ)
      (n := nsteps.toNat)
      (lo := lo)
      (step := step)
      (accum := default)
      (eff := (fun i _ => Fitree.trigger <| RegionE.RunRegion 0 []))
  | IOp.mk "scf.yield" vs [] 0 _ _ => .some do
      return (BlockResult.Ret vs)
  | _ => none

def handleScf: ScfE ~> Fitree PVoid :=
  fun _ e => nomatch e

instance: Semantics scf where
  E := ScfE
  semantics_op := scf_semantics_op
  handle := handleScf

/-
### Examples and testing
-/

namespace scf_if_true
def LHS (r1: Region scf) (r2: Region scf): Region scf := [mlir_region|
{
  "scf.if" (%b) ($(r1), $(r2)) : (i1) -> ()
}
]


def RHS (r1: Region scf) (r2: Region scf): Region scf := r1

-- | i1 true
def INPUT: SSAEnv arith := SSAEnv.One [⟨"x", MLIRType.i1, 0⟩]

def rhs (r1 r2: Region scf):
  Fitree (UBE +' SSAEnvE scf +' ScfE) (BlockResult scf) :=
  (Fitree.Vis
    (Sum.inr
      (Sum.inl
        (@SSAEnvE.Get _ _ _ scf (MLIRType.int Signedness.Signless 1)
          (@instInhabitedEval Void Void (fun _ => Unit) scf
            (@MLIR.AST.MLIRType.int Void Void (fun _ => Unit) scf .Signless 1))
          (SSAVal.SSAVal "b"))))
    fun
      (r : MLIRType.eval (α := Void) (MLIRType.int Signedness.Signless 1)) =>
    interp (M := Fitree (UBE +' SSAEnvE scf +' Semantics.E scf))
      (E := RegionE scf +' UBE +' Semantics.E scf)
      (fun (x : Type) (e : psum (RegionE scf) (UBE +' Semantics.E scf) x) =>
        (match x, e with
        | .(BlockResult scf), Sum.inl (RegionE.RunRegion i xs) =>
          List.get! (denoteRegions (ε' := fun x => Unit) scf [r1, r2]) i xs
        | x, Sum.inr (Sum.inl ube) => Fitree.Vis (Sum.inl ube) Fitree.ret
        | x, Sum.inr (Sum.inr se) => Fitree.Vis (Sum.inr (Sum.inr se)) Fitree.ret))
      (if (r == 0) = true
      then Fitree.Vis (Sum.inl (RegionE.RunRegion 0 [])) Fitree.ret
      else Fitree.Vis (Sum.inl (RegionE.RunRegion 1 [])) Fitree.ret))

theorem diff1 : @Fitree.Vis
      (psum.{0, 0, 0} UBE
        (psum.{0, 0, 0} (@SSAEnvE Void Void (fun (x : Void) => Unit) scf)
          (@Semantics.E Void Void (fun (x : Void) => Unit) scf instSemanticsVoidUnitScf)))  = 
    @Fitree.Vis (psum.{0, 0, 0} UBE (psum.{0, 0, 0} (@SSAEnvE Void Void (fun (x : Void) => Unit) scf) ScfE)) := by rfl

theorem diff2 : 
 (psum.{0, 0, 0} (@SSAEnvE Void Void (fun (x : Void) => Unit) scf)
          (@Semantics.E Void Void (fun (x : Void) => Unit) scf instSemanticsVoidUnitScf)
          (@MLIR.AST.MLIRType.eval Void Void (fun (x : Void) => Unit) scf
            (@MLIR.AST.MLIRType.int Void Void (fun (x : Void) => Unit) scf MLIR.AST.Signedness.Signless
              (@OfNat.ofNat.{0} Nat 1 (instOfNatNat 1))))) =
(psum.{0, 0, 0} (@SSAEnvE Void Void (fun (x : Void) => Unit) scf) ScfE
          (@MLIR.AST.MLIRType.eval Void Void (fun (x : Void) => Unit) scf
            (@MLIR.AST.MLIRType.int Void Void (fun (x : Void) => Unit) scf MLIR.AST.Signedness.Signless
              (@OfNat.ofNat.{0} Nat 1 (instOfNatNat 1))))) := by rfl


theorem diff3 : 
(@Semantics.E Void Void (fun (x : Void) => Unit) scf instSemanticsVoidUnitScf
            (@MLIR.AST.MLIRType.eval Void Void (fun (x : Void) => Unit) scf
              (@MLIR.AST.MLIRType.int Void Void (fun (x : Void) => Unit) scf MLIR.AST.Signedness.Signless
                (@OfNat.ofNat.{0} Nat 1 (instOfNatNat 1))))) = (ScfE
            (@MLIR.AST.MLIRType.eval Void Void (fun (x : Void) => Unit) scf
              (@MLIR.AST.MLIRType.int Void Void (fun (x : Void) => Unit) scf MLIR.AST.Signedness.Signless
                (@OfNat.ofNat.{0} Nat 1 (instOfNatNat 1))))) := by rfl 
  
theorem diff4: @denoteOp.match_2.{2} Void Void (fun (x : Void) => Unit) scf instSemanticsVoidUnitScf 
  = scf_if_true.rhs.match_1.{2} := by rfl 

set_option pp.all true in

theorem scf_if_sem:
  (denoteBBStmt (Δ := scf)
     (BasicBlockStmt.StmtOp
     (Op.mk "scf.if" [SSAVal.SSAVal "b"] [] [r1, r2] (AttrDict.mk [])
     (MLIRType.fn (MLIRType.tuple [MLIRType.int Signedness.Signless 1])
                  (MLIRType.tuple []))))) = rhs r1 r2
:= by {
  simp [denoteBBStmt, denoteOp, Semantics.semantics_op]
  simp_itree
  simp [scf_semantics_op];
  simp_itree;
  rewrite [diff4]; rfl;
}
#check scf_if_sem

/-
theorem LHS.sem (r1 r2: Region scf) (r: Option (BlockResult scf)) (env: SSAEnv scf):
    (run (denoteRegion _ (LHS r1 r2) []) INPUT) = (r, env) := by {
  simp [INPUT, LHS, run, denoteRegion, denoteBB, denoteBBStmts]
  simp [denoteBBStmt, denoteOp, Semantics.semantics_op];
  simp [scf_semantics_op];
  simp [List.zip];
  simp [List.zipWith];
  simp [List.mapM];
  simp_itree;
  simp [interp_ub]; simp_itree;
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get];
  simp;
  sorry
}
-/

/-
theorem equivalent (r1 r2: Region scf):
    (run (denoteRegion _ (LHS r1 r2) []) (INPUT)) =
    (run (denoteRegion _ (RHS r1 r2) []) (INPUT)) := by {
  simp [LHS, RHS, INPUT]
  simp [run, denoteRegion, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp]; simp_itree
  simp [interp_ub]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree

  repeat (simp [SSAEnv.get]; simp_itree)
  apply FinInt.xor_and
end scf_if_true
-/
namespace FOR_PEELING


theorem LHS (r: Region scf): Region scf := [mlir_region|
{
  "scf.for" (%c0, %cn_plus_1, %c1) ($(r)) : (i1) -> ()
}
]


theorem RHS  (r: Region scf): Region scf := [mlir_region|
{
  "scf.for" (%c0, %cn, %c1) ($(r)) : (i1) -> ()
  "scf.execute_region" () ($(r)) : ()
}]

theorem INPUT (n m: Nat): SSAEnv scf :=
    SSAEnv.One [⟨"cn", MLIRType.index, n⟩,
                ⟨"cn_plus_1", MLIRType.index, n + 1⟩,
                ⟨"c0", MLIRType.index, 0⟩,
                ⟨"c1", MLIRType.index, 1⟩]

theorem equivalent (n m: Nat) (r: Region scf):
    (run (denoteRegion _ (LHS r) []) (INPUT n m)) =
    (run (denoteRegion _ (RHS r) []) (INPUT n m)) := by sorry

end FOR_PEELING



namespace FOR_FUSION


theorem LHS (r: Region scf): Region scf := [mlir_region|
{
  "scf.for" (%c0, %cn, %c1) ($(r)) : (i1) -> ()
  "scf.for" (%cn, %cm, %c1) ($(r)) : (i1) -> ()
}
]


theorem RHS (r: Region scf): Region scf := [mlir_region|
{
  "scf.for" (%c0, %cn_plus_m, %c1) ($(r)) : (i1) -> ()
}]

theorem INPUT (n m: Nat): SSAEnv scf :=
    SSAEnv.One [⟨"cn", MLIRType.index, n⟩,
                ⟨"cm", MLIRType.index, m⟩,
                ⟨"cn_plus_m", MLIRType.index, n + m⟩,
                ⟨"c0", MLIRType.index, 0⟩,
                ⟨"c1", MLIRType.index, 1⟩]

theorem equivalent (n m: Nat) (r: Region scf):
    (run (denoteRegion _ (LHS r) []) (INPUT n m)) =
    (run (denoteRegion _ (RHS r) []) (INPUT n m)) := by sorry

end FOR_FUSION

