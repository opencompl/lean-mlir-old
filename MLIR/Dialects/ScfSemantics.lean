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
def run_loop_bounded_stepped_go [Monad m] (n: Nat) (i: Nat) (lo: Int) (step: Int)
  (accum: a) (eff: Int -> a -> m a): m a := do
   let ix : Int := lo + (n - i) * step
   let accum <- eff ix accum
   match i with
   | .zero => return accum
   | .succ i' => run_loop_bounded_stepped_go n i' lo step accum eff

-- | TODO: make this model the `yield` as well.
def run_loop_bounded_stepped [Monad m] (n: Nat) (lo: Int) (step: Int) (accum: a) (eff: Int -> a -> m a): m a :=
  run_loop_bounded_stepped_go n n lo step accum eff


-- | TODO: make this model the `yield` as well.
def run_loop_bounded
  (n: Nat)
  (ix: Nat)
  (start: BlockResult Δ):
    Fitree (RegionE Δ +' UBE +' ScfE) (BlockResult Δ) := do
  match n with
  | 0 => return start
  | .succ n' => do
    let new <- Fitree.trigger (RegionE.RunRegion 0 [⟨MLIRType.index, ix⟩])
    run_loop_bounded n' (ix + 1) new


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
    run_loop_bounded_stepped
      (a := BlockResult Δ)
      (n := nsteps.toNat)
      (lo := lo)
      (step := step)
      (accum := default)
      (eff := (fun i _ => Fitree.trigger <| RegionE.RunRegion 0 []))
  | IOp.mk "scf.for'" [⟨.index, lo⟩, ⟨.index, hi⟩] [] 1 _ _ => some do
      run_loop_bounded (n := hi - lo) (ix := 0) (BlockResult.Ret [])

  | IOp.mk "scf.yield" vs [] 0 _ _ => .some do
      return (BlockResult.Ret vs)
  | IOp.mk "scf.execute_region" args [] 1 _ _ => .some do
    Fitree.trigger (RegionE.RunRegion 0 args)
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

namespace SCF_IF_TRUE
def LHS (r1: Region scf) (r2: Region scf): Region scf := [mlir_region|
{
  "scf.if" (%b) ($(r1), $(r2)) : (i1) -> ()
}
]


def RHS (r1: Region scf) (r2: Region scf): Region scf := r1

-- | i1 true
def INPUT: SSAEnv arith := SSAEnv.One [⟨"x", MLIRType.i1, 0⟩]

def scfIfSem (r1 r2: Region scf):
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

theorem scf_if_sem:
  (denoteBBStmt (Δ := scf)
     (BasicBlockStmt.StmtOp
     (Op.mk "scf.if" [SSAVal.SSAVal "b"] [] [r1, r2] (AttrDict.mk [])
     (MLIRType.fn (MLIRType.tuple [MLIRType.int Signedness.Signless 1])
                  (MLIRType.tuple []))))) = scfIfSem r1 r2
:= by {
  simp [denoteBBStmt, denoteOp, Semantics.semantics_op]
  simp_itree
  simp [scf_semantics_op];
  simp_itree;
  rfl'; -- necessary to unfold through mutual inductives.
}

def LHS.semval (r1 r2: Region scf) := run
  (Fitree.Vis (Sum.inr (Sum.inl (SSAEnvE.Get (MLIRType.int Signedness.Signless 1) (SSAVal.SSAVal "b")))) fun r =>
      interp (M := Fitree (UBE +' SSAEnvE scf +' Semantics.E scf))  (E := RegionE scf +' UBE +' Semantics.E scf)
        (fun x e =>
          match x, e with
          | .(BlockResult scf), Sum.inl (RegionE.RunRegion i xs) => List.get! (denoteRegions scf [r1, r2]) i xs
          | x, Sum.inr (Sum.inl ube) => Fitree.Vis (Sum.inl ube) Fitree.ret
          | x, Sum.inr (Sum.inr se) => Fitree.Vis (Sum.inr (Sum.inr se)) Fitree.ret)
        (if (r == 0) = true then Fitree.Vis (Sum.inl (RegionE.RunRegion 0 [])) Fitree.ret
        else Fitree.Vis (Sum.inl (RegionE.RunRegion 1 [])) Fitree.ret))
    (SSAEnv.One [(SSAVal.SSAVal "x", { fst := MLIRType.i1, snd := 0 })])

theorem lhs_sem (r1 r2: Region scf):
    (run (denoteRegion _ (LHS r1 r2) []) INPUT) = LHS.semval r1 r2  := by {
  simp [LHS, INPUT, denoteRegion, denoteBB, denoteBBStmts];
  simp_itree;
  rewrite [scf_if_sem];
  rewrite [scfIfSem];
  simp_itree;
  rfl';
}

/-
(match
            Semantics.semantics_op
              (IOp.mk "scf.if" [{ fst := MLIRType.int Signedness.Signless 1, snd := default }] []
                (Nat.succ (Nat.succ 0)) (AttrDict.mk [])
                (MLIRType.fn (MLIRType.tuple [MLIRType.int Signedness.Signless 1]) (MLIRType.tuple []))) with

         -/
set_option maxHeartbeats 999999999 in
theorem equivalent (r1 r2: Region scf):
    (run (denoteRegion _ (LHS r1 r2) []) (INPUT)) =
    (run (denoteRegion _ (RHS r1 r2) []) (INPUT)) := by {
  simp [LHS, RHS, INPUT];
  simp [run, denoteRegion, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp]; simp_itree
  simp [interp_ub]; simp_itree
  simp [interp_ssa, interp_state, SSAEnvE.handle, SSAEnv.get]; simp_itree;
  simp [interp, Semantics.handle, handleScf];
  simp_itree;
  simp [Fitree.run];
  simp [Semantics.semantics_op];
  dsimp [scf_semantics_op];
  simp [interp];
  simp_itree;
  simp [interp];
  simp_itree;
  simp [denoteRegions];
  simp [List.get!];
  simp [Fitree_monad_right_identity];
}
#check equivalent


end SCF_IF_TRUE

namespace FOR_PEELING


theorem LHS (r: Region scf): Region scf := [mlir_region|
{
  "scf.for'" (%c0, %cn_plus_1) ($(r)) : (index, index) -> ()
}
]


theorem RHS  (r: Region scf): Region scf := [mlir_region|
{
  "scf.execute_region" (%c0) ($(r)) : (index) -> ()
  "scf.for'" (%c1, %cn_plus_1) ($(r)) : (index, index) -> ()
}]

theorem INPUT (n: Nat): SSAEnv scf :=
    SSAEnv.One [⟨"cn_plus_1", MLIRType.index, n + 1⟩,
                ⟨"c0", MLIRType.index, 0⟩,
                ⟨"c1", MLIRType.index, 1⟩]

set_option maxHeartbeats 999999999 in
theorem equivalent: ∀ (n: Nat) (r: Region scf),
    (run (denoteRegion _ (LHS r) []) (INPUT n)) =
    (run (denoteRegion _ (RHS r) []) (INPUT n)) := by {
  unfold INPUT, LHS, RHS;
  intros n;
  intros r;
  simp [denoteRegion, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp];
  simp_itree;
  simp [Semantics.semantics_op];
  simp [scf_semantics_op];
  simp [run_loop_bounded];
  simp [run];
  simp [interp_ub];
  simp [interp_ssa];
  simp [interp_state];
  simp [interp];
  simp_itree;
  simp [Semantics.handle];
  simp [handleScf];
  simp [Fitree.run];
  simp [SSAEnvE.handle]; simp [SSAEnv.get]; simp_itree;
  simp [SSAEnvE.handle]; simp [SSAEnv.get]; simp_itree;
  simp [denoteRegions];
  simp [List.get!];
  simp [denoteRegion];
  simp [Fitree.bind];
  simp [Fitree_monad_right_identity];
  simp [Fitree_bind_of_vis];
  simp [interp];
  simp_itree;
  -- rfl';
}



set_option maxHeartbeats 999999999 in
theorem equivalent: ∀ (n: Nat) (r: Region scf),
    (run (denoteRegion _ (LHS r) []) (INPUT n)) =
    (run (denoteRegion _ (RHS r) []) (INPUT n)) := by {
  unfold INPUT, LHS, RHS;
  intros n;
  induction n;
  case zero => {
    intros r;
    simp [denoteRegion, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp];
    simp_itree;
    simp [Semantics.semantics_op];
    simp [scf_semantics_op];
    simp[run];
    simp [interp_ub];
    simp_itree;
    simp [interp_ssa];
    simp [interp_state];
    simp_itree;
    simp [SSAEnvE.handle];
    simp[SSAEnv.get]; simp_itree;
    simp[SSAEnv.get]; simp_itree;
  }
  case succ n' H => {
    intros r;

    sorry; -- TODO: do the induction tomorrow.
  }
}

end FOR_PEELING



namespace FOR_FUSION


theorem LHS (r: Region scf): Region scf := [mlir_region|
{
  "scf.for'" (%c0, %cn) ($(r)) : (i32, i32) -> ()
  "scf.for'" (%cn, %cm) ($(r)) : (i32, i32) -> ()
}
]


theorem RHS (r: Region scf): Region scf := [mlir_region|
{
  "scf.for'" (%c0, %cn_plus_m) ($(r)) : (i32) -> ()
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

