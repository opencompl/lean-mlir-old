/-
## Dialect semantics

This file defines the interface that dialects provide to define their
semantics. This is built upon the `Dialect` interface from `MLIR.Dialects`
which define the custom attributes and type required to model the programs.
-/

import MLIR.Semantics.Fitree
import MLIR.Semantics.FitreeLaws
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Semantics.Dominance
import MLIR.AST
import MLIR.Util.Monads
open MLIR.AST


abbrev TypedArg (δ: Dialect α σ ε) := (τ: MLIRType δ) × MLIRType.eval τ


-- | Abbreviation with a typeclass context?
@[simp]
abbrev TypedArgs (δ: Dialect α σ ε) := List (TypedArg δ)


inductive OpM (Δ: Dialect α σ ϵ): Type -> Type _ where
| Ret: R -> OpM Δ R
| RunRegion: Nat -> TypedArgs Δ -> (TypedArgs Δ -> OpM Δ R) -> OpM Δ R
| Unhandled: String → OpM Δ R
| Error: String -> OpM Δ R


def OpM.map (f: A → B): OpM Δ A -> OpM Δ B
| .Ret a => .Ret (f a)
| .Unhandled s => .Unhandled s
| .RunRegion ix args k =>
    .RunRegion ix args (fun blockResult =>  (k blockResult).map f)
| .Error s => .Error s


def OpM.bind (ma: OpM Δ A) (a2mb: A -> OpM Δ B): OpM Δ B :=
  match ma with
  | .Unhandled s => .Unhandled s
  | .Ret a => a2mb a
  | .Error s => .Error s
  | .RunRegion ix args k =>
      .RunRegion ix args (fun blockResult => (k blockResult).bind a2mb)

instance : Monad (OpM Δ) where
   pure := OpM.Ret
   bind := OpM.bind

instance : LawfulMonad (OpM Δ) := sorry

-- Interpreted operation, like MLIR.AST.Op, but with less syntax
inductive IOp (δ: Dialect α σ ε) := | mk
  (name:    String) -- TODO: name should come from an Enum in δ.
  (resTy:   List (MLIRType δ))
  (args:    TypedArgs δ)
  (regions: List (TypedArgs δ → OpM δ (TypedArgs δ)))
  (attrs:   AttrDict δ)


-- The monad in which these computations are run
abbrev TopM (Δ: Dialect α σ ε) (R: Type _) := StateT (SSAEnv Δ) (Except (String × (SSAEnv Δ))) R
def TopM.run (t: TopM Δ R) (env: SSAEnv Δ): Except (String × (SSAEnv Δ)) (R × SSAEnv Δ) :=
  StateT.run t env

def TopM.scoped (t: TopM Δ R): TopM Δ R := do
  -- TODO: convert this to using the `SSAEnv`'s ability to a stack of `SSAScope`,
  -- instead of approximating it with overwriting the `SSAEnv` temporarily.
  let state ← get -- save scope
  let res ← t -- run computation
  set state -- restore scope
  return res -- return result


def TopM.raiseUB {Δ: Dialect α σ ε} (message: String): TopM Δ R := do
  let state ← get
  Except.error (message, state)

def TopM.get {Δ: Dialect α σ ε} (τ: MLIRType Δ) (name: SSAVal): TopM Δ τ.eval := do
  let s ← StateT.get
  match SSAEnv.get name τ s with
  | .some v => return v
  | .none => return default

def TopM.set {Δ: Dialect α σ ε} (τ: MLIRType Δ) (name: SSAVal) (v: τ.eval): TopM Δ Unit := do
  let s ← StateT.get
  match SSAEnv.get name τ s with
  | .some _ => TopM.raiseUB "setting to SSA value twice!"
  | .none => StateT.set (SSAEnv.set name τ v s)

theorem TopM.get_unfold {Δ: Dialect α σ ε} (τ: MLIRType Δ) (name: SSAVal) (env: SSAEnv Δ) :
    TopM.get τ name env =
    Except.ok (
      (match env.get name τ with
      | some v => v
      | none => default)
      , env) := by
  simp [TopM.get]
  simp_monad
  cases (env.get name τ) <;> rfl

theorem TopM.set_unfold {Δ: Dialect α σ ε} (τ: MLIRType Δ) (name: SSAVal) (env: SSAEnv Δ) (v: MLIRType.eval τ):
    TopM.set τ name v env =
    match env.get name τ with
    | some _ => Except.error ("setting to SSA value twice!", env)
    | none => Except.ok ((), env.set name τ v) := by
  simp [TopM.set]
  simp_monad
  cases (env.get name τ) <;> rfl

theorem TopM.set_ok {Δ: Dialect α σ ε} {τ: MLIRType Δ} {name: SSAVal} {v: MLIRType.eval τ} {env: SSAEnv Δ} {r}:
    TopM.set τ name v env = Except.ok r ->
    r = ((), env.set name τ v) ∧ env.get name τ = none := by 
  simp [set]; simp_monad
  cases (env.get name τ) <;> simp <;> intros H <;> try contradiction
  rw [H]

theorem TopM.get_env_set_commutes :
    TopM.get τ name env = Except.ok (r, env') ->
    name' ≠ name ->
    TopM.get τ name (env.set name' τ' v') = Except.ok (r, env'.set name' τ' v') := by
  intros H Hne
  rw [TopM.get_unfold] at *
  simp_monad at *
  simp_ssaenv at *
  revert H
  cases (env.get name τ) <;> simp at * <;> intros H <;> have ⟨H1, H2⟩ := H <;> subst r <;> subst env <;> simp

theorem TopM.set_env_set_commutes {Δ: Dialect α σ ε} (τ: MLIRType Δ) name v env r env':
    TopM.set τ name v env = Except.ok (r, env') ->
    ∀ name', name' ≠ name ->
    ∀ τ' v', ∃ env'', (env'.set name' τ' v').equiv env'' ∧
      TopM.set τ name v (env.set name' τ' v') = Except.ok (r, env'') := by
  intros H name' Hname τ' v'
  simp [set] at *; simp_monad at *
  revert H; cases Hget: (env.get name τ) <;> simp <;> intros H <;> try contradiction
  rw [SSAEnv.get_set_ne_val _ _ (by assumption)]
  rw [Hget]; simp
  subst env'
  exists (SSAEnv.set name τ v (SSAEnv.set name' τ' v' env))
  simp
  apply SSAEnv.set_commutes <;> try assumption
  simp [Ne.symm Hname]


theorem TopM.get_equiv {Δ: Dialect α σ ε} (env₁ env₂: SSAEnv Δ):
    env₁.equiv env₂ ->
    ∀ τ name r env₁', TopM.get τ name env₁ = Except.ok (r, env₁') →
                      TopM.get τ name env₂ = Except.ok (r, env₂):= by
  intros Hequiv τ name r env₁'
  repeat rw [TopM.get_unfold]
  rw [Hequiv]
  simp_monad
  intros H; have ⟨H, _⟩ := H
  rw [H]

theorem TopM.set_equiv {Δ: Dialect α σ ε} (env₁ env₂: SSAEnv Δ):
    env₁.equiv env₂ ->
    ∀ τ name v r env₁', TopM.set τ name v env₁ = Except.ok (r, env₁') →
      TopM.set τ name v env₂ = Except.ok (r, env₂.set name τ v):= by
  intros Hequiv τ name v r env₁'
  repeat rw [TopM.set_unfold]
  rw [Hequiv]
  simp_monad
  cases (SSAEnv.get name τ env₂) <;> simp

/-
TODO:

theorem raiseUB_commutes
theorem set_commutes_noninterfereing_set
theorem set_commutes_noninterfereing_get
theorem set_interference_ub
theorem set_commutes_set:
theorem get_commutes_get
theorem get_commutes_noninterfering_set
-/

class Semantics (Δ: Dialect α σ ε)  where
  -- Operation semantics function: maps an IOp (morally an Op but slightly less
  -- rich to guarantee good properties) to an interaction tree. Usually exposes
  -- any region calls then emits of event of E. This function runs on the
  -- program's entire dialect Δ but returns none for any operation that is not
  -- part of δ.
  -- TODO: make this such that it's a ddependent function, where we pass it the resTy and we expect
  -- an answer that matches the types of the resTy of the IOp.
  semantics_op: IOp Δ → OpM Δ (TypedArgs Δ)

-- This attribute allows matching explicit effect families like `ArithE`, which
-- often appear from bottom-up inference like in `Fitree.trigger`, with their
-- implicit form like `Semantics.E arith`, which often appears from top-down
-- type inference due to the type signatures in this module. Without this
-- attribute, instances of `Member` could not be derived, preventing most
-- lemmas about `Fitree.trigger` from applying.
-- attribute [reducible] Semantics.E

-- | TODO: Make this dependently typed to never allow such an error
def denoteTypedArgs (args: TypedArgs Δ) (names: List SSAVal): TopM Δ Unit :=
 match args with
 | [] => return ()
 | ⟨τ, val⟩::args =>
    match names with
    | [] => TopM.raiseUB "not enough names in denoteTypedArgs"
    | name :: names => do
        TopM.set τ name val
        denoteTypedArgs args names

-- Denote a region with an abstract `OpM.RunRegion`
def OpM.denoteRegion {Δ: Dialect α σ ε}
  (_r: Region Δ)
  (ix: Nat): TypedArgs Δ → OpM Δ (TypedArgs Δ) :=
   fun args => OpM.RunRegion ix args (fun retvals => OpM.Ret retvals)

-- Denote the list of regions with an abstract `OpM.runRegion`
def OpM.denoteRegions {Δ: Dialect α σ ε}
  (regions: List (Region Δ))
  (ix: Nat): List (TypedArgs Δ → OpM Δ (TypedArgs Δ)) :=
 match regions with
 | [] => []
 | r :: rs => (OpM.denoteRegion r ix) :: OpM.denoteRegions rs (ix + 1)

-- Denote a region by using its denotation function from the list
-- of regions. TODO: refactor to use Option
def TopM.denoteRegionsByIx
  (rs0: List (TypedArgs Δ → TopM Δ (TypedArgs Δ)))
 (ix: Nat) (args: TypedArgs Δ): TopM Δ (TypedArgs Δ) :=
  match rs0 with
  | [] => TopM.raiseUB s!"unknown region of ix {ix}"
  | r:: rs' =>
    match ix with
    | 0 => r args
    | ix' + 1 => TopM.denoteRegionsByIx rs' ix' args

-- Morphism from OpM to topM
def OpM.toTopM (rs0: List (TypedArgs Δ → TopM Δ (TypedArgs Δ))):
  OpM Δ (TypedArgs Δ) -> TopM Δ (TypedArgs Δ)
| OpM.Unhandled s => TopM.raiseUB s!"OpM.toTopM unhandled '{Δ.name}': {s}"
| OpM.Ret r => pure r
| OpM.Error s => TopM.raiseUB s
| OpM.RunRegion ix args k => do
       let ret <- TopM.denoteRegionsByIx rs0 ix args
       OpM.toTopM rs0 (k ret)
mutual
variable (Δ: Dialect α' σ' ε') [S: Semantics Δ]

-- unfolded version of List.map denoteRegion.
-- This allows the termination checker to view the termination.
def TopM.mapDenoteRegion:
  List (Region Δ) →
  List (TypedArgs Δ → TopM Δ (TypedArgs Δ))
| [] => []
| r :: rs => (TopM.scoped ∘ denoteRegion r) :: TopM.mapDenoteRegion rs

def denoteOpArgs (args: List (TypedSSAVal Δ)) : TopM Δ (List (TypedArg Δ)) := do
  args.mapM (fun (name, τ) => do
        pure ⟨τ, ← TopM.get τ name⟩)

-- Convert a region to its denotation to establish finiteness.
-- Then use this finiteness condition to evaluate region semantics.
-- Use the morphism from OpM to TopM.
def denoteOp (op: Op Δ):
    TopM Δ (TypedArgs Δ) :=
  match op with
  | .mk name res0 args0 regions0 attrs => do
      let resTy := res0.map Prod.snd
      let args ← denoteOpArgs args0
      -- Built the interpreted operation
      let iop : IOp Δ := IOp.mk name resTy args (OpM.denoteRegions regions0 0) attrs
      -- Use the dialect-provided semantics, and substitute regions
      let ret ← OpM.toTopM (TopM.mapDenoteRegion regions0) (S.semantics_op iop)
      match res0 with
      | [] => pure ()
      | [res] => match ret with
          | [⟨τ, v⟩] => TopM.set τ res.fst v
          | _ => TopM.raiseUB s!"denoteOp: expected 1 return value, got '{ret}'"
      | _ => TopM.raiseUB s!"denoteOp: expected 0 or 1 results, got '{res0}'"
      return ret
  -- denote a sequence of ops
def denoteOps (stmts: List (Op Δ)): TopM Δ (TypedArgs Δ) :=
   match stmts with
   | [] => return  []
   | [stmt] => denoteOp stmt
   | (stmt :: stmts') => do
        let _ ← denoteOp stmt
        denoteOps stmts'

def denoteRegion (rgn: Region Δ) (args: TypedArgs Δ):
    TopM Δ (TypedArgs Δ) := do
  match rgn with
  | Region.mk name formalArgsAndTypes ops =>
     -- TODO: check that types in [TypedArgs] is equal to types at [bb.args]
     -- TODO: Any checks on the BlockResults of intermediate ops?
     let formalArgs : List SSAVal := formalArgsAndTypes.map Prod.fst
     denoteTypedArgs args formalArgs
     denoteOps ops


end

section Retraction

variable {α₁ σ₁ ε₁} {δ₁: Dialect α₁ σ₁ ε₁}
 variable   {α₂ σ₂ ε₂} {δ₂: Dialect α₂ σ₂ ε₂}

-- TODO: create holes for things that are unknown? eg. use `undefined?
def MLIRType.retractLeft: MLIRType (δ₁ + δ₂) → MLIRType δ₁
| .int sgn sz => .int sgn sz -- : Signedness -> Nat -> MLIRType δ
| .float sz => .float sz -- : Nat -> MLIRType δ
| .index => .index --:  MLIRType δ
| .tensor1d => .tensor1d
| .tensor2d => .tensor2d
| .tensor4d => .tensor4d
| .erased => .erased
| .undefined s => .undefined s-- : String → MLIRType δ
| .extended (Sum.inl σ₁) => .extended σ₁ -- : σ → MLIRType δ
| .extended (Sum.inr σ₂) => .erased

def MLIRType.swapDialect: MLIRType (δ₁ + δ₂) -> MLIRType (δ₂ + δ₁)
| .int sgn sz => (.int sgn sz) -- : Signedness -> Nat -> MLIRType δ
| .float sz => (.float sz) -- : Nat -> MLIRType δ
| .index => (.index) --:  MLIRType δ
| .erased => .erased
| .tensor1d => .tensor1d
| .tensor2d => .tensor2d
| .tensor4d => .tensor4d
| .undefined s => (.undefined s) -- : String → MLIRType δ
| .extended (Sum.inl σ₁) => .extended (Sum.inr σ₁)
| .extended (Sum.inr σ₂) => .extended (Sum.inl σ₂)


def TypedArg.swapDialect: TypedArg (δ₁ + δ₂) -> TypedArg (δ₂ + δ₁)
| ⟨.int sgn sz, v ⟩ =>  ⟨ .int sgn sz, v ⟩ -- : Signedness -> Nat -> MLIRType δ
| ⟨ .float sz, v ⟩ =>  ⟨.float sz, v ⟩ -- : Nat -> MLIRType δ
| ⟨.index, v⟩ => ⟨.index, v ⟩ --:  MLIRType δ
| ⟨.undefined s, v ⟩ =>  ⟨.undefined s, v⟩ -- : String → MLIRType δ

| ⟨.tensor1d, v⟩ => ⟨.tensor1d, v ⟩
| ⟨.tensor2d, v⟩ => ⟨.tensor2d, v ⟩
| ⟨.tensor4d, v⟩ => ⟨.tensor4d, v ⟩
| ⟨.extended (Sum.inl σ₁), v ⟩ => ⟨.extended (Sum.inr σ₁), v⟩
| ⟨.extended (Sum.inr σ₂), v ⟩ => ⟨.extended (Sum.inl σ₂), v⟩
| ⟨.erased, ()⟩ => ⟨.erased, ()⟩

@[reducible, simp]
def TypedArgs.swapDialect (ts: TypedArgs (δ₁ + δ₂)): TypedArgs (δ₂ + δ₁) :=
  ts.map TypedArg.swapDialect



def TypedArg.retractLeft (t: TypedArg (δ₁ + δ₂)):  TypedArg δ₁ :=
match t with
| ⟨.int sgn sz, v ⟩ =>  ⟨ .int sgn sz, v ⟩ -- : Signedness -> Nat -> MLIRType δ
| ⟨ .float sz, v ⟩ => ⟨.float sz, v ⟩ -- : Nat -> MLIRType δ
| ⟨.index, v⟩ => ⟨.index, v ⟩ --:  MLIRType δ
| ⟨.tensor1d, v⟩ =>  ⟨.tensor1d,v ⟩
| ⟨.tensor2d, v⟩ =>  ⟨.tensor2d,v ⟩
| ⟨.tensor4d, v⟩ =>  ⟨.tensor4d,v ⟩
| ⟨.undefined s, v ⟩ =>  ⟨.undefined s, v⟩ -- : String → MLIRType δ
| ⟨.extended (Sum.inl σ₁), v ⟩ =>  ⟨.extended σ₁, v⟩ -- : σ → MLIRType δ
| ⟨.extended (Sum.inr σ₂), v ⟩ => ⟨.erased, () ⟩
| ⟨.erased, ()⟩ =>  ⟨.erased, ()⟩

def TypedArg.retractRight (t: TypedArg (δ₁ + δ₂)):  TypedArg δ₂ :=
match t with
| ⟨.int sgn sz, v ⟩ =>  ⟨ .int sgn sz, v ⟩ -- : Signedness -> Nat -> MLIRType δ
| ⟨ .float sz, v ⟩ => ⟨.float sz, v ⟩ -- : Nat -> MLIRType δ
| ⟨.index, v⟩ => ⟨.index, v ⟩ --:  MLIRType δ
| ⟨.tensor1d, v⟩ =>  ⟨.tensor1d,v ⟩
| ⟨.tensor2d, v⟩ =>  ⟨.tensor2d,v ⟩
| ⟨.tensor4d, v⟩ =>  ⟨.tensor4d,v ⟩
| ⟨.undefined s, v ⟩ =>  ⟨.undefined s, v⟩ -- : String → MLIRType δ
| ⟨.extended (Sum.inl σ₁), v ⟩ =>  ⟨.erased, ()⟩ -- : σ → MLIRType δ
| ⟨.extended (Sum.inr σ₂), v ⟩ => ⟨.extended σ₂, v ⟩
| ⟨.erased, ()⟩ =>  ⟨.erased, ()⟩


@[reducible, simp]
def TypedArgs.retractLeft (ts: TypedArgs (δ₁ + δ₂)): TypedArgs δ₁ :=
  ts.map TypedArg.retractLeft

@[reducible, simp]
def TypedArgs.retractRight (ts: TypedArgs (δ₁ + δ₂)): TypedArgs δ₂ :=
  ts.map TypedArg.retractRight

-- TODO: define the attribute dictionary retraction.
-- Will need to rectact over entries, which will need a retraction over values.
mutual
def AttrValues.retractLeft: List (AttrValue  (δ₁ + δ₂)) -> List (AttrValue δ₁)
| [] => []
| a::as => a.retractLeft:: AttrValues.retractLeft as

def MLIR.AST.AttrValue.retractLeft: AttrValue (δ₁ + δ₂) -> AttrValue δ₁
| .symbol s => .symbol s
| .permutation p => .permutation p
| .nat n => .nat n
| .str s => .str s
| .int i t => .int i (MLIRType.retractLeft t)
| .bool b => .bool b
| .float f t => .float f (MLIRType.retractLeft t)
| .type t => .type (MLIRType.retractLeft t)
| .affine aff => .affine aff
| .list as => .list <| AttrValues.retractLeft as
| .extended (.inl x) => .extended x
| .extended (.inr _) => .erased
| .erased => .erased
| .opaque_ dialect value => .opaque_ dialect value
| .opaqueElements dialect value ty => .opaqueElements dialect value .erased
| .unit => .unit
| .dict d => .dict <| d.retractLeft
| .alias x => .alias x
| .nestedsymbol x y => .nestedsymbol x.retractLeft y.retractLeft


def MLIR.AST.AttrEntry.retractLeft: AttrEntry (δ₁ + δ₂) -> AttrEntry δ₁
| .mk k v => .mk k v.retractLeft

def AttrEntries.retractLeft: List (AttrEntry (δ₁ + δ₂)) -> List (AttrEntry δ₁)
| [] => []
| e :: es => e.retractLeft :: AttrEntries.retractLeft es

def MLIR.AST.AttrDict.retractLeft: AttrDict (δ₁ + δ₂) -> AttrDict δ₁
| .mk es => AttrDict.mk (AttrEntries.retractLeft es)
end

-- Retract right
mutual
def AttrValues.swapDialect: List (AttrValue  (δ₁ + δ₂)) -> List (AttrValue (δ₂  + δ₁))
| [] => []
| a::as => a.swapDialect:: AttrValues.swapDialect as

def MLIR.AST.AttrValue.swapDialect: AttrValue (δ₁ + δ₂) -> AttrValue (δ₂ + δ₁)
| .symbol s => .symbol s
| .permutation p => .permutation p
| .nat n => .nat n
| .str s => .str s
| .int i t => .int i (MLIRType.swapDialect t)
| .bool b => .bool b
| .float f t => .float f (MLIRType.swapDialect t)
| .type t => .type (MLIRType.swapDialect t)
| .affine aff => .affine aff
| .list as => .list <| AttrValues.swapDialect as
| .extended (.inl x) => .extended (.inr x)
| .extended (.inr x) => .extended (.inl x)
| .erased => .erased
| .opaque_ dialect value => .opaque_ dialect value
| .opaqueElements dialect value ty => .opaqueElements dialect value .erased
| .unit => .unit
| .dict d => .dict <| d.swapDialect
| .alias x => .alias x
| .nestedsymbol x y => .nestedsymbol x.swapDialect y.swapDialect


def MLIR.AST.AttrEntry.swapDialect: AttrEntry (δ₁ + δ₂) -> AttrEntry (δ₂ + δ₁)
| .mk k v => .mk k v.swapDialect

def AttrEntries.swapDialect: List (AttrEntry (δ₁ + δ₂)) -> List (AttrEntry (δ₂ + δ₁))
| [] => []
| e :: es => e.swapDialect :: AttrEntries.swapDialect es

def MLIR.AST.AttrDict.swapDialect: AttrDict (δ₁ + δ₂) -> AttrDict (δ₂ + δ₁)
| .mk es => AttrDict.mk (AttrEntries.swapDialect es)
end

def OpM.swapDialect: OpM (δ₁ + δ₂) (TypedArgs (δ₁ + δ₂)) -> OpM (δ₂ + δ₁) (TypedArgs (δ₁ + δ₂))
| OpM.Ret r => OpM.Ret r
| OpM.Unhandled s => OpM.Unhandled s
| OpM.Error s => OpM.Error s
| OpM.RunRegion ix args k =>
  OpM.RunRegion ix (TypedArgs.swapDialect args) (fun retargs =>
              OpM.swapDialect (k (TypedArgs.swapDialect retargs)))

def IOp.swapDialect: IOp (δ₁ + δ₂) -> IOp (δ₂ + δ₁)
| IOp.mk  (name:    String) -- TODO: name should come from an Enum in δ.
  (resTy:   List (MLIRType (δ₁ + δ₂)))
  (args:    TypedArgs (δ₁ + δ₂))
  (regions: List (TypedArgs (δ₁ + δ₂) -> OpM (δ₁ + δ₂) (TypedArgs (δ₁ + δ₂))))
  (attrs:   AttrDict (δ₁ + δ₂)) =>
     IOp.mk name
        (resTy.map MLIRType.swapDialect)
        (args.map TypedArg.swapDialect)
        (AttrDict.swapDialect attrs)
        -- conjugate region by swapping dialect.
        (regions := regions.map  (fun rgnEff => (fun args =>
                 (rgnEff (TypedArgs.swapDialect args)).swapDialect.map TypedArgs.swapDialect)))

-- a -> a + b
def TypedArg.injectLeft: TypedArg (δ₁) -> TypedArg (δ₁ +  δ₂)
| ⟨.int sgn sz, v ⟩ =>  ⟨ .int sgn sz, v ⟩ -- : Signedness -> Nat -> MLIRType δ
| ⟨ .float sz, v ⟩ =>  ⟨.float sz, v ⟩ -- : Nat -> MLIRType δ
| ⟨.index, v⟩ => ⟨.index, v ⟩ --:  MLIRType δ
| ⟨.undefined s, v ⟩ =>  ⟨.undefined s, v⟩ -- : String → MLIRType δ
| ⟨.tensor1d, v⟩ => ⟨.tensor1d, v ⟩
| ⟨.tensor2d, v⟩ => ⟨.tensor2d, v ⟩
| ⟨.tensor4d, v⟩ => ⟨.tensor4d, v ⟩
| ⟨.extended σ, v ⟩ => ⟨.extended (Sum.inl σ), v⟩
| ⟨.erased, ()⟩ => ⟨.erased, ()⟩


@[reducible, simp]
def TypedArg.injectRight: TypedArg δ₂ -> TypedArg (δ₁ + δ₂) :=
  TypedArg.swapDialect ∘ TypedArg.injectLeft

@[reducible, simp]
def TypedArgs.injectLeft (ts: TypedArgs (δ₁)): TypedArgs (δ₁ + δ₂) :=
  ts.map TypedArg.injectLeft

@[reducible, simp]
def TypedArgs.injectRight (ts: TypedArgs (δ₂)): TypedArgs (δ₁ + δ₂) :=
  ts.map TypedArg.injectRight

def OpM.retractLeft [Inhabited R]: OpM (δ₁+ δ₂) R -> OpM δ₁  R
| OpM.Error s => OpM.Error s
| OpM.Unhandled s => OpM.Unhandled s
| OpM.Ret r => OpM.Ret r
| OpM.RunRegion ix args k =>
  OpM.RunRegion ix args.retractLeft (fun results => (k results.injectLeft).retractLeft)

-- Retract an IOp to the left component.
-- TODO: IOp needs to be profunctorial, region can use more stuff than the operation
-- strictly has?
def IOp.retractLeft: IOp (δ₁ + δ₂) -> IOp δ₁
| IOp.mk  (name:    String) -- TODO: name should come from an Enum in δ.
  (resTys:   List (MLIRType (δ₁ + δ₂)))
  (args:    TypedArgs (δ₁ + δ₂))
  (regions: List (TypedArgs (δ₁ + δ₂) -> OpM (δ₁+δ₂) (TypedArgs (δ₁ + δ₂))))
  (attrs:   AttrDict (δ₁ + δ₂)) =>
  let resTys' := resTys.map MLIRType.retractLeft
  let args' := args.map TypedArg.retractLeft
  let attrs' := AttrDict.retractLeft attrs
  let regions' := regions.map (fun rgnEff =>
    (fun args => (rgnEff args.injectLeft).retractLeft.map TypedArgs.retractLeft ))
  (IOp.mk name resTys' args' regions' attrs')

def IOp.retractRight (op: IOp (δ₁ + δ₂)): IOp δ₂ :=
  IOp.retractLeft (IOp.swapDialect op)

def OpM.injectLeft: OpM δ₁ (TypedArgs δ₁) -> OpM (δ₁ + δ₂) (TypedArgs (δ₁ + δ₂))
| OpM.Ret r => OpM.Ret r.injectLeft
| OpM.Error s => OpM.Error s
| OpM.Unhandled s => OpM.Unhandled s
| OpM.RunRegion ix args k =>
  OpM.RunRegion ix args.injectLeft (fun args => (k args.retractLeft).injectLeft)

@[simp, reducible]
def OpM.injectRight: OpM δ₂ (TypedArgs δ₂) -> OpM (δ₁ + δ₂) (TypedArgs (δ₁ + δ₂))
| OpM.Ret r => OpM.Ret r.injectRight
| OpM.Error s => OpM.Error s
| OpM.Unhandled s => OpM.Unhandled s
| OpM.RunRegion ix args k =>
  OpM.RunRegion ix args.injectRight (fun args => (k args.retractRight).injectRight)



-- Or the two OpM, using unhandled as the unit for the or.
def OpM.orUnhandled: OpM δ₁ (TypedArgs δ₁)
  -> OpM δ₂ (TypedArgs δ₂) -> OpM (δ₁ + δ₂) (TypedArgs (δ₁ + δ₂))
| OpM.Error e, _ => OpM.Error e
| _, OpM.Error e => OpM.Error e
| OpM.Unhandled x, OpM.Unhandled y => OpM.Unhandled s!"(({δ₁.name}) ({x}) | ({δ₂.name}) ({y}))"
| OpM.Unhandled _, x => x.injectRight
| x, _ => x.injectLeft



-- TODO: Allow the semantics to be defined in such a way that a dialect like `scf`
-- can successfully 'forward' extended type arguments.
instance
    {α₁ σ₁ ε₁} {δ₁: Dialect α₁ σ₁ ε₁}
    {α₂ σ₂ ε₂} {δ₂: Dialect α₂ σ₂ ε₂}
    [S₁: Semantics δ₁]
    [S₂: Semantics δ₂]
    : Semantics (δ₁ + δ₂) where
  -- semantics_op: IOp Δ → Fitree (RegionE Δ +' UBE) (BlockResult Δ)
  semantics_op op :=
    let op₁ := IOp.retractLeft op
    let op₂ := IOp.retractRight op
    let res1 :=  (S₁.semantics_op op₁)
    let res2 :=  (S₂.semantics_op op₂)
    OpM.orUnhandled res1 res2



def run! {Δ: Dialect α' σ' ε'}  {R} [Inhabited R]
    (t: TopM Δ R) (env: SSAEnv Δ):
    R × SSAEnv Δ :=
   match t.run env with
   | .error err => panic! s!"error when running progam: {err}"
   | .ok val => val

def run {Δ: Dialect α' σ' ε'} {R}
    (t: TopM  Δ R) (env: SSAEnv Δ):
    Except (String × SSAEnv Δ) (R × SSAEnv Δ) :=
  StateT.run t env

-- The property for two programs to execute with no error and satisfy a
-- post-condition
def semanticPostCondition₂ {Δ: Dialect α' σ' ε'}
    (t₁ t₂: Except String (R × SSAEnv Δ))
    (f: R → SSAEnv Δ → R → SSAEnv Δ → Prop) :=
  match t₁, t₂ with
  | .ok (r₁, env₁), .ok (r₂, env₂) => f r₁ env₁ r₂ env₂
  | _, _ => False

@[simp] theorem semanticPostCondition₂_ok_ok:
  semanticPostCondition₂ (Except.ok (r₁, env₁)) (Except.ok (r₂, env₂)) f =
  f r₁ env₁ r₂ env₂ := rfl

/-
### Denotation notation
-/

class Denote (δ: Dialect α σ ε) [S: Semantics δ]
    (T: {α σ: Type} → {ε: σ → Type} → Dialect α σ ε → Type) where
  denote: T δ → TopM δ (TypedArgs δ)

notation "⟦ " t " ⟧" => Denote.denote t

instance DenoteOp (δ: Dialect α σ ε) [Semantics δ]: Denote δ Op where
  denote op := denoteOp δ op
-- This only works for single-BB regions with no arguments
instance DenoteRegion (δ: Dialect α σ ε) [Semantics δ]: Denote δ Region where
  denote r := denoteRegion δ r []

-- Not for regions because we need to specify the fuel

@[simp] theorem Denote.denoteOp [Semantics δ]:
  Denote.denote (self := DenoteOp δ) op = denoteOp δ op := rfl
@[simp] theorem Denote.denoteRegion [Semantics δ]:
  Denote.denote (self := DenoteRegion δ) r = denoteRegion δ r [] := rfl


/-
### PostSSAEnv

A PostSSAEnv is a predicate on an environment, that check that it can be the
resulting environment of the interpretation of a TopM monad.
-/

macro "simp_semantics_monad" : tactic =>   
  `(tactic| simp_monad;
            (repeat rw [TopM.get_unfold]);
            (repeat rw [TopM.get_unfold]); simp)

macro "simp_semantics_monad" "at" Hname:ident : tactic =>   
  `(tactic| simp_monad at $Hname;
            (repeat rw [TopM.get_unfold] at $Hname:ident);
            (repeat rw [TopM.set_unfold] at $Hname:ident);
            simp at $Hname:ident)

macro "simp_semantics_monad" "at" "*" : tactic =>   
  `(tactic| simp_monad at *;
            (repeat rw [TopM.get_unfold] at *);
            (repeat rw [TopM.set_unfold] at *);
            simp at *)

def postSSAEnv (m: TopM δ R) (env: SSAEnv δ) : Prop :=
  ∃ env' v, run m env' = .ok (v, env)


theorem denoteOpArgs_env_set_preserves [S: Semantics Δ]
    (args: List (TypedSSAVal Δ)):
    ∀ env r resEnv,
    denoteOpArgs Δ args env = Except.ok (r, resEnv) ->
    ∀ name, args.all (fun arg => name ≠ arg.fst) ->
    ∀ τ v, denoteOpArgs Δ args (env.set name τ v) = Except.ok (r, resEnv.set name τ v) := by
  induction args
  case nil =>
    intros env r resEnv H
    simp [denoteOpArgs] at *
    simp_monad at *
    cases H; subst r env; simp
  case cons head tail HInd =>
    intros env r resEnv H name Hname τ v
    simp [denoteOpArgs]; simp [denoteOpArgs] at H
    have ⟨headName, headτ⟩ := head
    simp_monad at *
    cases Hhead: (TopM.get headτ headName env)
    case error _ =>
      rw [Hhead] at H; contradiction
    case ok headRes =>
      rw [Hhead] at H
      simp [List.all, List.foldr] at Hname
      have ⟨Hname_head, _⟩ := Hname
      rw [TopM.get_env_set_commutes Hhead Hname_head]
      simp at *
      split at H <;> try contradiction
      case h_2 rTail Htail =>
      have ⟨rTailRes, rTailEnv⟩ := rTail
      simp at H; have ⟨_, _⟩ := H; subst r resEnv
      simp [denoteOpArgs] at *
      simp [bind, StateT.bind, Except.bind, pure, StateT.pure, Except.pure] at HInd
      rw [HInd] <;> assumption

theorem denoteTypedArgs_cons_args {δ: Dialect α σ ε} {argsHead: TypedArg δ}
    {argsTail: TypedArgs δ} {vals: List SSAVal} {env: SSAEnv δ} {res} {env': SSAEnv δ} :
  denoteTypedArgs (argsHead::argsTail) vals env = Except.ok (res, env') →
  ∃ valHead valTail,
    vals = valHead::valTail ∧
    TopM.set argsHead.fst valHead argsHead.snd env =
      Except.ok ((), env.set valHead argsHead.fst argsHead.snd) ∧ 
    denoteTypedArgs argsTail valTail (env.set valHead argsHead.fst argsHead.snd) =
      Except.ok ((), env'):= by
  intros H
  simp [denoteTypedArgs] at H
  cases vals <;> try contradiction
  case cons headVal tailVal =>
  exists headVal
  exists tailVal
  simp_monad at *
  revert H
  split <;> intros H <;> try contradiction
  case h_2 v Hv =>
  have ⟨fst, snd⟩ := v; cases fst
  case unit =>
  simp_semantics_monad at *
  revert Hv; split <;> intros Hv <;> try contradiction
  simp at Hv; subst snd
  simp [H]


def denoteTypedArgs_cons_unfold (headArgs: TypedArg Δ) (tailArgs: List (TypedArg Δ)) (env: SSAEnv Δ):
    denoteTypedArgs (headArgs::tailArgs) (headVal::tailVal) env =
      (do
         TopM.set headArgs.fst headVal headArgs.snd
         denoteTypedArgs tailArgs tailVal) env := by
  simp [denoteTypedArgs]


def denoteTypedArgs_equiv {Δ: Dialect α σ ε} {args: TypedArgs Δ} :
    ∀ {vals} {env₁} {r} {env₁'}, 
    denoteTypedArgs args vals env₁ = Except.ok (r, env₁') →
    ∀ {env₂}, env₁.equiv env₂ →
    ∃ env₂', denoteTypedArgs args vals env₂ = Except.ok (r, env₂') ∧
             env₁'.equiv env₂' := by
  induction args
  case nil =>
    intros val env₁ r env₁' H env₂ Henv₂
    simp [denoteTypedArgs] at *
    exists env₂
    simp_monad at *; subst H; assumption
  case cons argsHead argsTail HInd =>
    intros vals env₁ r env₁' H env₂ Henv₂
    have ⟨valsHead, valsTail, HVals, HHead, HTail⟩ := denoteTypedArgs_cons_args H
    subst vals
    have HHead₂ := TopM.set_equiv _ _ Henv₂ _ _ _ _ _ HHead
    specialize (HInd HTail (SSAEnv.equiv_set _ _ Henv₂ _ _ _))
    have ⟨env₂', Henv₂', Hequiv₂'⟩ := HInd
    exists env₂'
    cases r
    rw [denoteTypedArgs_cons_unfold]
    simp_monad
    rw [HHead₂]; simp; rw [Henv₂']
    simp; assumption


def run_denoteTypedArgs_env_set_preserves [S: Semantics Δ] (regArgs: TypedArgs Δ):
    ∀ vals env res env',
    denoteTypedArgs regArgs vals env = Except.ok (res, env') →
    ∀ name, name ∉ vals →
    ∀ τ v, ∃ env'',
    denoteTypedArgs regArgs vals (SSAEnv.set name τ v env) = Except.ok (res, env'') ∧
    (SSAEnv.set name τ v env').equiv env'' := by
  induction regArgs
  case nil =>
    intros vals env res env' H name _ τ v
    simp [denoteTypedArgs] at *; simp_monad at *; subst env
    exists (env'.set name τ v)
    simp
    apply SSAEnv.equiv_rfl
  case cons head tail HInd =>
    intros vals env res env' H name Hname τ v
    have ⟨valHead, valTail, HVal, HHead, HTail⟩ := denoteTypedArgs_cons_args H
    subst vals; have ⟨HNameHead, HNameTail⟩ := List.ne_mem_cons _ _ _ Hname
    rw [denoteTypedArgs_cons_unfold]; simp_monad
    have ⟨envHead, HEnvHeadEquiv, HEnvHead⟩ := TopM.set_env_set_commutes _ _ _ _ _ _ HHead _ HNameHead τ v
    have ⟨HEnvHead, _⟩ := TopM.set_ok HEnvHead
    simp at HEnvHead; subst envHead
    have ⟨env'', HdenoteTail, HEquiv⟩ := HInd _ _ _ _ HTail name HNameTail τ v
    have ⟨env₂'', Henv₂'', Henv₂equiv''⟩ := denoteTypedArgs_equiv HdenoteTail _ HEnvHeadEquiv
    exists env₂''
    simp [HEnvHead, Henv₂'']
    apply SSAEnv.equiv_trans _ _ (by assumption) _ (by assumption)


def run_denoteRegionByIx_env_set_preserves {Δ: Dialect α σ ε} [S: Semantics Δ] 
  (regions: List (TypedArgs Δ -> TopM Δ (TypedArgs Δ))) :
    ∀ name τ v,
    (∀ region args env res env', region ∈ regions ->
      region args env = Except.ok (res, env') ->
      region args (env.set name τ v)  = Except.ok (res, env'.set name τ v)
      ) ->
    ∀ idx args env res env', 
    TopM.denoteRegionsByIx regions idx args env = Except.ok (res, env') ->
    TopM.denoteRegionsByIx regions idx args (env.set name τ v) =
      Except.ok (res, env'.set name τ v) := by 
  -- We just find the region we have to run, and apply the recursion
  unfold TopM.denoteRegionsByIx
  cases regions <;> simp <;> intros name τ v H idx args env res env' Hrun <;> try contradiction
  case cons head tail =>    
    cases idx
    case zero =>
      simp at *
      specialize (H head args env res env' (by constructor) Hrun)
      assumption
    case succ idx' =>
      simp at *
      let Hind := (run_denoteRegionByIx_env_set_preserves tail)
      specialize (Hind name τ v)
      specialize (Hind (by 
        intros region args env res env' Hregions Hrun
        specialize (H region args env res env' (by constructor; assumption) Hrun)
        assumption
      ))
      specialize (Hind idx' args env res env' Hrun)
      assumption


def OpM.toTopM_regions_env_set_preserves {Δ: Dialect α σ ε} [S: Semantics Δ]
  (regions: List (TypedArgs Δ -> TopM Δ (TypedArgs Δ))) :
    ∀ name τ v,
    (∀ region args env res env', region ∈ regions ->
      region args env = Except.ok (res, env') ->
      region args (env.set name τ v)  = Except.ok (res, env'.set name τ v)
      ) ->
    ∀ opM env res env',
    OpM.toTopM regions opM env = Except.ok (res, env') -> 
    OpM.toTopM regions opM (env.set name τ v) = Except.ok (res, (env'.set name τ v)) := by
  intros name τ v Hregs opM env res env' H
  cases opM <;> try contradiction

  -- Ret case, we return the same value in both cases, so this is trivial
  case Ret ret =>
    unfold OpM.toTopM; unfold OpM.toTopM at H
    cases H <;> simp
    rfl

  -- Running a region. This is the inductive case over opM
  case RunRegion idx args continuation =>
    unfold OpM.toTopM; unfold OpM.toTopM at H
    have ⟨⟨resReg, envReg⟩, HReg⟩ := ExceptMonad.split H
    simp [Bind.bind, StateT.bind, Except.bind] at *
    rw [HReg] at H; simp at H

    have HdenoteIx := @run_denoteRegionByIx_env_set_preserves
    specialize (HdenoteIx regions name τ v Hregs idx args env _ _ HReg)
    simp [HdenoteIx]
    apply OpM.toTopM_regions_env_set_preserves <;> assumption

def TopM.set_env_set_preserves :
  TopM.set τ name v env = Except.ok (r, env') ->
  TopM.set τ name v (env.set name' τ' v') = Except.ok (r, env'.set name' τ' v') := by sorry



mutual
variable {Δ: Dialect α σ ε} [S: Semantics Δ] (name: SSAVal) (τ: MLIRType Δ) (v: MLIRType.eval τ)

-- run_denoteOp_env_set_preserves
--   mapDenoteRegion_env_set_preserves
--     denoteRegion_env_set_preserves
--        run_denoteOps_env_set_preserves
--          run_denoteOp_env_set_preserves
def run_denoteOp_env_set_preserves: ∀ (op: Op Δ),
    ∀ env r env',
    denoteOp Δ op env = Except.ok (r, env') ->
    denoteOp Δ op (SSAEnv.set name τ v env) =
      Except.ok (r, SSAEnv.set name τ v env')
  | Op.mk op_name res args regions attrs => by
    -- unfold the denotation
    unfold denoteOp; simp

    simp [bind, StateT.bind, Except.bind]
    intro env r env' H

    -- Take care of the arguments
    split at H <;> try contradiction
    case h_2 _ argsRes HargsRes =>
    rw [denoteOpArgs_env_set_preserves HargsRes]
    simp
    -- Take care of the regions
    split at H <;> try contradiction
    case h_2 _ opR HopRes =>
    have ⟨opR, opEnv'⟩ := opR
    have Hind := @OpM.toTopM_regions_env_set_preserves
    specialize (@Hind _ _ _ _ _ (TopM.mapDenoteRegion Δ regions) name τ v)
    specialize (@Hind (mapDenoteRegion_env_set_preserves regions))
    rw [Hind _ _ _ _ HopRes]
    simp
    cases res
    case nil => simp at *; cases H; rfl
    case cons headRes tailRes =>
      cases tailRes
      case cons _ _ => simp at *; cases H 
      case nil =>
        simp 
        cases opR
        case nil => simp at *; cases H 
        case cons opRHead opRTail => 
          cases opRTail
          case cons _ _ =>  simp at *; cases H
          case nil => 
              simp at *
              split at H <;> try contradiction
              case h_2 setRes HSetRes =>
              rw [TopM.set_env_set_preserves HSetRes]
              simp; cases H; rfl

def run_denoteOps_env_set_preserves :
    ∀ (ops: List (Op Δ)) env res env',
    denoteOps Δ ops env = Except.ok (res, env') ->
    denoteOps Δ ops (SSAEnv.set name τ v env) = Except.ok (res, SSAEnv.set name τ v env')
  | [] => by
    unfold denoteOps
    simp; intros _ _ _ H
    cases H <;> rfl
  | head::tail => by
    unfold denoteOps
    match TAIL:tail with
    | .nil =>
        have HIndOp := @run_denoteOp_env_set_preserves head
        apply HIndOp
    | .cons head2 tail2 =>
        have HIndOp := @run_denoteOp_env_set_preserves head
        intros _ _ _ H
        have ⟨⟨res ,env''⟩, Hhead⟩ := ExceptMonad.split H
        simp [bind, StateT.bind, Except.bind] at *
        simp [HIndOp _ _ _ Hhead]
        rw [Hhead] at H; simp at H
        rw[<- TAIL];
        apply (run_denoteOps_env_set_preserves tail);
        rw[TAIL]; assumption;


def denoteRegion_env_set_preserves: ∀ region,
    ∀ args env res env',
    denoteRegion Δ region args env = Except.ok (res, env') ->
    denoteRegion Δ region args (env.set name τ v) = Except.ok (res, env'.set name τ v)
  | Region.mk reg_name regArgs ops => by
    unfold denoteRegion; simp
    simp [bind]; simp [StateT.bind]
    intros args env res env' H
    cases Hargs: (denoteTypedArgs args (List.map Prod.fst regArgs) env)
    case error e =>
      rw [Hargs] at H
      contradiction
    case ok discr =>
      have ⟨_, discr⟩ := discr
      rw [Hargs] at H
      simp [bind, Except.bind] at H
      rw [run_denoteTypedArgs_env_set_preserves Hargs]
      simp [bind, Except.bind]
      apply (run_denoteOps_env_set_preserves ops)
      apply H

def mapDenoteRegion_env_set_preserves:
  ∀ regions region args env res env', region ∈ (TopM.mapDenoteRegion Δ regions) ->
    region args env = Except.ok (res, env') ->
    region args (env.set name τ v)  = Except.ok (res, env'.set name τ v)
  | [] => by
    intros _ _ _ _ _ _
    contradiction
  | head::tail => by
    intros region args env res env' HregIn Hreg
    simp [TopM.mapDenoteRegion] at HregIn
    cases HregIn
    case head =>
      simp [TopM.scoped, Function.comp, bind, StateT.bind, Except.bind] at *
      simp [get, getThe, MonadStateOf.get, StateT.get, pure, Except.pure] at *
      split at Hreg <;> try contradiction
      case h_2 _ regRes Hreg' =>
      have ⟨regRes, regResEnv⟩ := regRes
      rw [@denoteRegion_env_set_preserves head] <;> try assumption
      simp [set, StateT.set, pure, Except.pure, StateT.pure] at *
      cases Hreg; subst env regRes
      simp
    case tail =>
      apply (mapDenoteRegion_env_set_preserves tail) <;> assumption
end




def postSSAEnv.env_set_preserves [S: Semantics δ] (op: Op δ) (env: SSAEnv δ) :
    postSSAEnv ⟦op⟧ env →
    ∀ name, isVarFreeInOp name op ->
    postSSAEnv ⟦op⟧ (env.set name τ v) := by
  sorry