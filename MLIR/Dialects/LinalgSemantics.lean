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

-- | the denotation of the affine map is indexes that
-- tell us
def DenoteAffineMap := Nat × Nat

-- verify that an affine tuple has two components
def liftA2_Option (a?: Option A) (b?: Option B): Option (A × B) :=
  match a? with
  | .some a => match b? with | .some b => .some (a, b) | _ => .none
  | .none => .none


-- | index the tuple (nat, nat) with the index. This is used to
-- generate the semantics of the affine_map.
def index_tuple (t: Nat × Nat) (ix: Nat): Option Nat :=
  match ix with
  | 0 => .some t.fst
  | 1 => .some t.snd
  | _ => none

-- | TODO: generalize tuple of arguments to list
-- | return which index to use.
def rhs_affine_expr_to_var (l: String) (r: String) (v: String): Option (Nat) :=
  if v == l
  then .some (0)
  else if v == r
  then .some (1)
  else .none

-- | affine tuple must be of the form (a, b) -> (c, d)
def verify_2d_to_2d_affine_map (aff_map: AffineMap): Option (Nat × Nat) :=
  match aff_map with
  | AffineMap.mk
      (AffineTuple.mk [AffineExpr.Var i1, AffineExpr.Var i2])
      (AffineTuple.mk [AffineExpr.Var o1, AffineExpr.Var o2])  =>
          liftA2_Option (rhs_affine_expr_to_var i1 i2 o1) (rhs_affine_expr_to_var i1 i2 o2)
  | _ => none



-- TODO @lephe: could you perform the mutual induction please?
def MLIRTy_eval_equal_after_coe [Δ: Dialect α σ ε] (τ: MLIRTy):
    τ.eval = (coeMLIRType (c := CoeDialectEmpty (δ := Δ)) τ).eval := sorry

-- TODO @lephe: another proof obligation
def MLIRType_builtin_eval_equal_after_coe [Δ: Dialect α σ ε] [coe: CoeDialect builtin Δ] (τ: MLIRType builtin):
    τ.eval = (coeMLIRType (c := coe) τ).eval := sorry

-- | compose the index (ix0, ix1) with the permutation that is (affine_map) to arrive
-- at the final indexing operation.
def tensor_index (d0 d1: Nat) (ix0 ix1: Nat) (affine_map: Nat × Nat): Option Nat := do
  let ixs := (ix0, ix1)
  let ix0' <- (index_tuple affine_map 0) >>= (index_tuple ixs)
  let ix1' <- (index_tuple affine_map 1) >>= (index_tuple ixs)
  return (ix0' * d1 + ix1')

-- TODO: how do I write the semantics for this in a way that
-- I can get access to the `tensor` type?
def linalg_parallel_iter [Δ: Dialect α σ ε]
   (d0 d1: Nat)
   (inTensor:  Matrix d0 d1 τ)
   (ix0 ix1: Nat)
   (affine_map: Nat × Nat): Fitree ((RegionE Δ) +' UBE +' LinalgE) (Option τ.eval) := do
  -- | lol, have fun reasoning with this...
  let data? := (tensor_index d0 d1 ix0 ix1 affine_map) >>= inTensor.data.get?
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
   (d0 d1: Nat)
   (inTensor: Matrix d0 d1 τ)
   (affine_map: Nat × Nat):
     Fitree ((RegionE Δ) +' UBE +' LinalgE) (TypedArgs Δ) := do
  -- | TODO: Yeesh, we gotta worry about List.bind.
  let ixs : List (Nat × Nat) := (List.range d0).bind (fun ix1 => (List.range d1).map (fun ix2 => (ix1, ix2)))
  let outValues <- ixs.mapM (fun ix2d => linalg_parallel_iter d0 d1 inTensor ix2d.fst ix2d.snd affine_map)
  let outValues := list_option_to_option_list outValues
  match outValues with
  | .some outValues =>
        -- | TODO:
        let t : Tensor τ:= { inTensor.toTensor with data := outValues, h_data_size := sorry }
        let dims : DimList :=  [Dimension.Known d0, Dimension.Known  d1]
        let out_tensor_τ := builtin.tensor dims τ
        let out_tensor := RankedTensor.mk (D := dims) (toTensor := t) sorry
        return [⟨out_tensor_τ, MLIRType_builtin_eval_equal_after_coe out_tensor_τ ▸ out_tensor⟩]
  | .none => do
      Fitree.trigger $ UBE.DebugUB "RankedTensor: unable to produce output args."
      return []


-- def toy_semantics_op (ret_name: Option SSAVal) (op: Op builtin):
-- | TODO: we need a way to say that `builtin` is a member of Gδ
-- @lephe: do you want me to thread the dialect projection everywhere?
def linalg_semantics_op  [CoeDialect builtin Δ] [P: DialectProjection Δ builtin]: IOp Δ →
      Option (Fitree (RegionE Δ +' UBE +' LinalgE) (BlockResult Δ))
  | IOp.mk "linalg.parallel2d1" [⟨.extended sΔ, v⟩] [] 1 attrs _ =>
        match AttrDict.find attrs "indexing_maps" with
        | some (.affine affine_map?) =>
          match (verify_2d_to_2d_affine_map affine_map?) with
          | .some affine_map =>
              match H: DialectProjection.project_σ (self := P) _ _ sΔ with
              | some (builtin.σ.tensor [Dimension.Known d0, Dimension.Known d1] τ) => .some do
                  let input: RankedTensor [Dimension.Known  d0, Dimension.Known  d1] τ :=
                    cast (by rw [H]) <| DialectProjection.project_ε (self := P) sΔ v
                  let out  <- linalg_parallel_all_iters d0 d1 input affine_map
                  return (BlockResult.Ret out)
              | _ => none
          | _ => none
        | _ => none
  | IOp.mk "linalg.parallel1d2" [input1, input2] [] 1 _ _ => some do
      sorry
  | _ => none


instance: Semantics Linalg where
  E := fun T => Void
  semantics_op := linalg_semantics_op
  handle T voidT := nomatch voidT



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



namespace TRANSPOSE_INVOLUTION



theorem LHS:Region builtin := [mlir_region|
{
  %t = "linalg.generic" (%input) ({
    ^entry(%v: i32):
      "return" (%v) : () -> i32
  }) : tensor<2×4×i32> -> tensor<4×2×i32>
  %out = "linalg.generic" (%t) ({
    ^entry(%v: i32):
      "return" (%v) : () -> i32
  }) : tensor<4×2×i32> -> tensor<2×4×i32>
}]




theorem INPUT (input: RankedTensor [.Known 2, .Known 4] .i32): SSAEnv builtin :=
      [[("input", ⟨builtin.tensor [.Known 2, .Known 4] .i32, input⟩)]]

set_option maxHeartbeats 999999999 in
theorem equivalent (input: RankedTensor [.Known 2, .Known 4] .i32) (r: Region scf):
    ((run (denoteRegion _ LHS []) (INPUT input)) |>.snd.get "out" (builtin.tensor [.Known 2, .Known 4] .i32)) = .some input := by {
  unfold INPUT, LHS;
  simp [denoteRegion, denoteBB, denoteBBStmts, denoteBBStmt, denoteOp];
  simp_itree;

}

end TRANSPOSE_INVOLUTION
