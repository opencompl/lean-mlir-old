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
    let (_: BlockResult Δ) <- Fitree.trigger (RegionE.RunRegion 0 [⟨MLIRType.index, ix⟩])
    run_loop_bounded n' (ix + 1) (BlockResult.Ret [])


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
      run_loop_bounded (n := hi - lo) (ix := lo) (BlockResult.Ret [])

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

set_option pp.rawOnError true in
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
}
#check equivalent


end SCF_IF_TRUE

section FITREE_THEOREMS

private def optionT_defaultHandler2: E ~> OptionT (Fitree E) :=
  fun _ e => OptionT.lift $ Fitree.trigger e

theorem interp_ub_Vis:
    interp_ub (Fitree.Vis e k: Fitree (UBE +' E) R) =
    interp (Fitree.case_ UBE.handle optionT_defaultHandler2) (.Vis e k) := by
  rfl

private def stateT_defaultHandler2: E ~> StateT (SSAEnv δ) (Fitree E) :=
  fun _ e => StateT.lift $ Fitree.trigger e

theorem interp_ssa_Vis:
    interp_ssa (Fitree.Vis e k: Fitree (SSAEnvE δ +' E) R ) =
    interp (Fitree.case_ SSAEnvE.handle stateT_defaultHandler2) (.Vis e k) := by
  rfl


theorem interp_Vis {E M} [Monad M] (h: E ~> M) {T} (e: E T) (k: T → Fitree E R):
    interp h (Fitree.Vis e k) =
    bind (h _ e) (fun x => interp h (k x)) := by
  rfl

/-
def run {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {R}
    (t: Fitree (UBE +' SSAEnvE Δ +' S.E) R) (env: SSAEnv Δ):
    Option R × SSAEnv Δ :=
-/

theorem run_fequal {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {T R}
    (k k': T → Fitree (UBE +' SSAEnvE Δ +' S.E) R) e env
    interm_val interm_env:
    run (.Vis e .Ret) env = (some interm_val, interm_env) →
    (run (k interm_val) interm_env = run (k' interm_val) interm_env) →
    run (.Vis e k) env = run (.Vis e k') env := by
  intros h
  cases e <;> simp
  simp [run] at *
  simp [interp_ub] at *
  repeat sorry

set_option pp.notation false in
theorem run_Vis {Δ: Dialect α' σ' ε'} [S: Semantics Δ]
  (k: T → Fitree (UBE +' SSAEnvE Δ +' S.E) R) e env
  interm_val interm_env:
    run (.Vis e .Ret) env = (some interm_val, interm_env) →
    run (.Vis e k) env = run (k interm_val) interm_env := by {
      intros RUN;
      simp [run];
      simp [run] at RUN;
      simp [interp_ssa, interp_ub] at *;
      simp [Semantics.handle, interp_state] at *;
      simp [interp] at *;
      simp [bind];
      simp [OptionT.bind, OptionT.mk];
      sorry
  }

theorem run_SSAEnvE_get
  [Δ: Dialect α σ ε]
  [S: Semantics Δ]
  (name: String)
  (τ: MLIRType Δ)
  (v: τ.eval)
  (env: SSAEnv Δ)
  (ENV: SSAEnv.get (SSAVal.SSAVal name) τ env = some v)
  (k: τ.eval → Fitree (UBE +' SSAEnvE Δ +' S.E) R):
  run (Fitree.Vis (Sum.inr (Sum.inl (SSAEnvE.Get τ (SSAVal.SSAVal name)))) fun x => k x) env = run (k v) env := by {
      simp [run];
      simp [interp, interp_ssa, interp_state, interp_ub];
      simp_itree;
      simp [SSAEnvE.handle]
      simp [ENV];
      rfl;
  }

/-
  match
    run
      (denoteOp_interp_region scf (denoteRegions scf [r]) (BlockResult scf)
        (Sum.inl (RegionE.RunRegion 0 [{ fst := MLIRType.index, snd := 0 }])))
      (INPUT n) with
  | (none, env') => (none, env')
  | (some x, env') =>
  -/

#check denoteOp_interp_region
theorem   run_denoteOp_interp_region
  [Δ: Dialect α σ ε]
  [S: Semantics Δ]
  (rs: List (Region Δ))
  (run_ix: Nat)
  (run_args: TypedArgs Δ)
  (env: SSAEnv Δ):
    run (denoteOp_interp_region Δ (denoteRegions Δ rs) (BlockResult Δ)
        (Sum.inl (RegionE.RunRegion run_ix run_args))) env = run (List.get! (denoteRegions Δ rs) run_ix run_args) env := by {
        simp [interp_ub, denoteOp_interp_region, denoteRegions];
}

-- this one is written using (denoteRegion Δ r) instead of (denoteRegions Δ rs)
theorem   run_denoteOp_interp_region'
  [Δ: Dialect α σ ε]
  [S: Semantics Δ]
  (denotes: List (TypedArgs Δ → Fitree (UBE +' SSAEnvE Δ +' S.E) (BlockResult Δ)))
  (run_ix: Nat)
  (run_args: TypedArgs Δ)
  (env: SSAEnv Δ):
    run (denoteOp_interp_region Δ denotes (BlockResult Δ)
        (Sum.inl (RegionE.RunRegion run_ix run_args))) env = run (List.get! denotes run_ix run_args) env := by {
        simp [interp_ub, denoteOp_interp_region, denoteRegions];
}

theorem run_denoteOp_interp_region''
  [Δ: Dialect α σ ε]
  [S: Semantics Δ]
  (denotes: List (TypedArgs Δ → Fitree (UBE +' SSAEnvE Δ +' S.E) (BlockResult Δ)))
  (run_ix: Nat)
  (run_args: TypedArgs Δ)
  (env: SSAEnv Δ):
    run (interp (denoteOp_interp_region scf (denoteRegions scf [r])) (run_loop_bounded n 0 (BlockResult.Ret [])))
      = rhs := by {
          simp [denoteRegions];
          simp [denoteOp_interp_region];
          simp [List.get!];

}



theorem bind_fequal [Monad M] [LawfulMonad M] {A B: Type}
  (lhs: M A) (rhs1 rhs2: A -> M B) (rhs_equal: ∀ (a: A), rhs1 a = rhs2 a):
    lhs >>= rhs1 = lhs >>= rhs2 := by {
      suffices (rhs1 = rhs2) by {
        simp [this];
      }
      funext x;
      simp [rhs_equal];
}

-- interp (Fitree.case_ SSAEnvE.handle stateT_defaultHandler) (k r)

set_option pp.notation false in
theorem interp_ssa_bind [Monad M] [LawfulMonad M]
  (t: Fitree (SSAEnvE δ +' E) A)
  (k: A -> Fitree (SSAEnvE δ +' E)  B):
  interp_ssa (Fitree.bind t k) = StateT.bind (interp_ssa t) (fun a => interp_ssa (k a)) := by {
  induction t;
  case Ret lawful r => {
    simp [interp, bind, Fitree.bind, StateT.bind, interp_ssa, interp_state];
  }
  case Vis lawful T' e' k' IND => {
    simp [interp_ssa, interp, bind, Fitree.bind, StateT.bind, interp_state] at *;
    funext s;
    rw [Fitree_monad_assoc];
    simp[IND];
  }
}
theorem run_bind {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {T R}
    t (k: T → Fitree (UBE +' SSAEnvE Δ +' S.E) R) env:
  run (Fitree.bind t k) env =
    match run t env with
    | (none, env') => (none, env')
    | (some x, env') => run (k x) env' := by
  simp [run, bind]
  unfold interp_ub
  simp [interp_bind, bind, OptionT.bind, OptionT.mk]
  simp [interp_ssa, interp_state]
  simp [interp_bind, bind, StateT.bind]
  simp [Fitree.run_bind]
  sorry
end FITREE_THEOREMS


namespace FOR_PEELING


theorem LHS (r: Region scf): Region scf := [mlir_region|
{
  "scf.for'" (%c0, %cn) ($(r)) : (index, index) -> ()
}
]


theorem RHS  (r: Region scf): Region scf := [mlir_region|
{
  "scf.execute_region" (%c0) ($(r)) : (index) -> ()
  "scf.for'" (%c1, %cn) ($(r)) : (index, index) -> ()
}]

theorem INPUT (n: Nat): SSAEnv scf :=
    SSAEnv.One [⟨"cn_plus_1", MLIRType.index, n + 1⟩,
                ⟨"c0", MLIRType.index, 0⟩,
                ⟨"c1", MLIRType.index, 1⟩]


theorem CORRECT_r (n:Nat) (r: Region scf) args:
    exists x,
    (run (denoteRegion scf r args) (INPUT n)) = (some x, INPUT n) := by
  sorry

/-
run
    (Fitree.bind
      (denoteOp_interp_region scf [denoteRegion scf r] (BlockResult scf)
        (Sum.inl (RegionE.RunRegion 0 [{ fst := MLIRType.index, snd := 0 }])))
      fun discr =>
      Fitree.Vis (Sum.inr (Sum.inl (SSAEnvE.Get MLIRType.index (SSAVal.SSAVal "c1")))) fun r_1 =>
        Fitree.Vis (Sum.inr (Sum.inl (SSAEnvE.Get MLIRType.index (SSAVal.SSAVal "cn_plus_1")))) fun r_2 =>
          interp (denoteOp_interp_region scf [denoteRegion scf r])
-/
-- when a region obeys CORRECT_r, it is safe to commute the
-- read from memory with a run of the region:
theorem CORRECT_r_commute_run_denoteOp_interp_region_SSAEnvE_get
  [S: Semantics scf]
  (r: Region scf)
  (args: TypedArgs scf)
  (CORRECT_r: (run (denoteRegion scf r args) (INPUT n)) = (some x, INPUT n))
  (discr: BlockResult scf)
  (τ: MLIRType scf)
  (v: MLIRType.eval τ)
  (name: String)
  (ENV: SSAEnv.get (SSAVal.SSAVal name) τ (INPUT n) = some v)
  (k: BlockResult scf → τ.eval → Fitree (UBE +' SSAEnvE scf +' Semantics.E scf) R):
  run
      (Fitree.bind
        (denoteOp_interp_region scf [denoteRegion scf r] (BlockResult scf)
          (Sum.inl (RegionE.RunRegion 0 args)))
        (fun discr =>
          Fitree.Vis (Sum.inr (Sum.inl (SSAEnvE.Get τ (SSAVal.SSAVal name)))) fun v => k discr v)) (INPUT n) =
    run
          (Fitree.Vis (Sum.inr (Sum.inl (SSAEnvE.Get τ (SSAVal.SSAVal name)))) fun v =>
            (Fitree.bind (denoteOp_interp_region scf [denoteRegion scf r] (BlockResult scf) (Sum.inl (RegionE.RunRegion 0 args)))
                (fun discr => k discr v))) (INPUT n)

    := by {
      rewrite [run_bind];
      simp [denoteOp_interp_region];
      simp [List.get!];
      simp [CORRECT_r];
      simp [run_Vis];
      simp_itree;
      rewrite [run_SSAEnvE_get (name := name) (τ := τ) (v := v) (ENV := ENV)];
      rewrite [run_SSAEnvE_get (name := name) (τ := τ) (v := v) (ENV := ENV)];
      rewrite [run_bind];
      simp [CORRECT_r];
  }

/-

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
-/

theorem peel_run_loop_bounded {n ix: Nat} (start: BlockResult Δ):
    run_loop_bounded (n+1) ix start =
    bind (Fitree.trigger (RegionE.RunRegion 0 [⟨.index, ix⟩]))
      (fun (_: BlockResult Δ) =>
        run_loop_bounded n (ix+1) (BlockResult.Ret [])) := by
  simp [run_loop_bounded]

theorem equivalent: ∀ (n m: Nat) (r: Region scf),
    n = Nat.succ m →
    (run (denoteRegion _ (LHS r) []) (INPUT n)) =
    (run (denoteRegion _ (RHS r) []) (INPUT n)) := by {
  unfold LHS, RHS;
  intros n m r h_n_m;
  simp [denoteRegion, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp];
  simp_itree;
  simp [Semantics.semantics_op];
  simp [scf_semantics_op];
  rewrite [run_SSAEnvE_get (name := "c0") (τ := MLIRType.index) (v := 0)];
  rewrite [run_SSAEnvE_get (name := "cn") (τ := MLIRType.index) (v := n)];

  rewrite [run_SSAEnvE_get (name := "c0") (τ := MLIRType.index) (v := 0)];
  simp [denoteRegions];
  simp_itree;
  simp [Fitree_monad_right_identity];
  rewrite [CORRECT_r_commute_run_denoteOp_interp_region_SSAEnvE_get];

  rewrite [run_SSAEnvE_get (name := "c1") (τ := MLIRType.index) (v := 1)];
  rewrite [CORRECT_r_commute_run_denoteOp_interp_region_SSAEnvE_get];
  rewrite [run_SSAEnvE_get (name := "cn") (τ := MLIRType.index) (v := n)];

  simp [h_n_m, (by sorry: Nat.succ m - 1 = m)];
  rw [peel_run_loop_bounded];
  simp_itree;

  /-
  rewrite [run_denoteOp_interp_region'];
  -- TODO: commute memory read again.
  rewrite [CORRECT_r_commute_run_denoteOp_interp_region_SSAEnvE_get];

  simp [run_bind];
  -- TODO: try to commute the vis with the bind?


  simp [interp]; dsimp_itree
  rw [run_SSAEnvE_get (name := "cn_plus_1") (τ := MLIRType.index) (v := n + 1)]
  simp [];
  rewrite [run_denoteOp_interp_region' (Δ := scf) (denotes:= [denoteRegion scf r]) (run_ix := 0)]
  -- | PLEASE rewrite under binders jesus christ
  -- rewrite [run_SSAEnvE_get (name := "c1") (τ := MLIRType.index) (v := 1) (env := env')]
  induction n; -- is this dubious?
  simp [run_loop_bounded];
  simp_itree;
  simp [run_bind];
  rewrite [run_denoteOp_interp_region'  (Δ := scf) (denotes:= [denoteRegion scf r]) (run_ix := 0)];
  simp [denoteRegions];
  simp [List.get!];
  rewrite [I_HATE_YOU];
  simp;

  simp [run]
  simp [interp_ub_Vis];
  simp [interp_Vis];
  simp [bind, OptionT.bind, OptionT.mk];

  simp [interp_bind]; dsimp_itree;
  simp [interp_ssa_Vis]; dsimp_itree; unfold INPUT; simp [SSAEnvE.handle, INPUT, SSAEnv.get];
  simp [pure];
  dsimp_itree;
  simp [denoteRegions];
  simp [interp_bind];
  dsimp_itree;
  simp [interp];
  simp [denoteRegion];
  simp [List.get!];
  -/

/-  have h (k: Nat → Fitree (UBE +' SSAEnvE scf +' ScfE) (BlockResult scf)) :=
    run_Vis (Δ := scf) k (Sum.inr (Sum.inl (SSAEnvE.Get MLIRType.index (SSAVal.SSAVal "c1")))) (INPUT n) 1 (INPUT n);
  rw [h] -/
  rfl
  done
}


/-
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
-/
end FOR_PEELING



namespace FOR_FUSION

set_option pp.notation false in
theorem interp_region_of_run_loop_bounded
(r: Region scf)
(n: ℕ)
(rhs: Fitree (UBE +' SSAEnvE scf +' Semantics.E scf) (BlockResult scf)):
   (interp (denoteOp_interp_region scf (denoteRegions scf [r])) (run_loop_bounded n 0 (BlockResult.Ret []))) = rhs := by {
    induction n;
    case zero => {
      simp [run_loop_bounded];
      simp [interp];
      sorry
    }
    case succ n' => {
      simp [run_loop_bounded];
      simp [interp];
      simp[List.cons];
    }
   }

theorem LHS (r: Region scf): Region scf := [mlir_region|
{
  "scf.for'" (%c0, %cn) ($(r)) : (index, index) -> ()
  "scf.for'" (%cn, %cm) ($(r)) : (index, index) -> ()
}
]


theorem RHS (r: Region scf): Region scf := [mlir_region|
{
  "scf.for'" (%c0, %cn_plus_m) ($(r)) : (index, index) -> ()
}]

def fitreeExpand (f: Fitree E R) (h: (T: Type) -> E T ->  Fitree F T): Fitree F R :=
 match f with
 | .Ret r => .Ret r
 | .Vis e k => Fitree.bind (h _ e) (fun t => fitreeExpand (k t) h)

/-
theorem interp_of_fitree_is_bind {M} [Monad M] {E} (h: E ~>  Fitree F)
    (f: Fitree E R): interp (h := h) f = Fitree.translate h f := by sorry
-/

theorem INPUT (n m: Nat): SSAEnv scf :=
    SSAEnv.One [⟨"cn", MLIRType.index, n⟩,
                ⟨"cm", MLIRType.index, m⟩,
                ⟨"cn_plus_m", MLIRType.index, n + m⟩,
                ⟨"c0", MLIRType.index, 0⟩,
                ⟨"c1", MLIRType.index, 1⟩]

theorem equivalent (n m: Nat) (r: Region scf):
    (run (denoteRegion _ (LHS r) []) (INPUT n m)) =
    (run (denoteRegion _ (RHS r) []) (INPUT n m)) := by {
  unfold INPUT, LHS, RHS;
  simp [denoteRegion, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp];
  simp_itree;
  simp [run_Vis];
  rewrite [run_SSAEnvE_get (name := "c0") (τ := MLIRType.index) (v := 0)];
  rewrite [run_SSAEnvE_get (name := "cn") (τ := MLIRType.index) (v := n)];
  simp [Semantics.semantics_op];
  simp [scf_semantics_op];

  simp [run_bind]
  simp [run_denoteOp_interp_region];
  rewrite [run_SSAEnvE_get (name := "c0") (τ := MLIRType.index) (v := 0)];
  rewrite [run_SSAEnvE_get (name := "cn_plus_m") (τ := MLIRType.index) (v := n + m)];
  simp;
}

end FOR_FUSION

