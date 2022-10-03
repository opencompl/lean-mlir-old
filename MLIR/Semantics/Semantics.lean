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
import MLIR.AST
open MLIR.AST


abbrev TypedArg (δ: Dialect α σ ε) := (τ: MLIRType δ) × MLIRType.eval τ


-- | Abbreviation with a typeclass context?
@[simp]
abbrev TypedArgs (δ: Dialect α σ ε) := List (TypedArg δ)


inductive OpM (Δ: Dialect α σ ϵ): Type -> Type _ where
| Ret: R -> OpM Δ R
/- We could have had RunRegion take a `Region` as a parameter.
  This would mean that a user could, in theory, craft a request to run a region
  to our main loop, and we would happily do so!
  This means that the framework is capable of expressing *more* than MLIR.
  It is capable of expressing of operations that 'JIT' regions during the execution.

  Ah ha, but we can't do that. If we try, the termination checker complains
  that our function isn't well terminating!
-/
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

-- Interpreted operation, like MLIR.AST.Op, but with less syntax
inductive IOp (δ: Dialect α σ ε) := | mk
  (name:    String) -- TODO: name should come from an Enum in δ.
  (resTy:   List (MLIRType δ))
  (args:    TypedArgs δ)
  (regions: List (TypedArgs δ → OpM δ (TypedArgs δ))) -- TODO: surely, I can build the denotation of a region and pass it along to you?
  (attrs:   AttrDict δ)


-- The monad in which these computations are run
abbrev TopM (Δ: Dialect α σ ε) (R: Type _) := StateT (SSAEnv Δ) (Except String) R

def TopM.raiseUB {Δ: Dialect α σ ε} (message: String): TopM Δ R :=
  Except.error message

def TopM.get {Δ: Dialect α σ ε} (τ: MLIRType Δ) (name: SSAVal): TopM Δ τ.eval :=
  sorry

def TopM.set {Δ: Dialect α σ ε} (τ: MLIRType Δ) (name: SSAVal) (v: τ.eval): TopM Δ Unit :=
  sorry


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

-- | TODO: throw error if we don't have enough names
-- | TODO: Make this dependently typed to never allow such an error
def denoteTypedArgs (args: TypedArgs Δ) (names: List SSAVal): TopM Δ Unit :=
 match args with
 | [] => return ()
 | ⟨τ, val⟩::args =>
    match names with
    | [] => return ()
    | name :: names => do
        TopM.set τ name val

-- Denote a region with an abstract `OpM.RunRegion`
def denoteRegionOpM {Δ: Dialect α σ ε}
  (_r: Region Δ)
  (ix: Nat): TypedArgs Δ → OpM Δ (TypedArgs Δ) :=
   fun args => OpM.RunRegion ix args (fun retvals => OpM.Ret retvals)

-- Denote the list of regions with an abstract `OpM.runRegion`
def denoteRegionsOpM {Δ: Dialect α σ ε}
  (regions: List (Region Δ))
  (ix: Nat): List (TypedArgs Δ → OpM Δ (TypedArgs Δ)) :=
 match regions with
 | [] => []
 | r :: rs => (denoteRegionOpM r ix) :: denoteRegionsOpM rs (ix + 1)

-- Denote a region by using its denotation function from the list
-- of regions. TODO: refactor to use Option
def denoteRegionByIx
  (rs0: List (TypedArgs Δ → TopM Δ (TypedArgs Δ)))
 (ix: Nat) (args: TypedArgs Δ): TopM Δ (TypedArgs Δ) :=
  match rs0 with
  | [] => TopM.raiseUB s!"unknown region of ix {ix}"
  | r:: rs' =>
    match ix with
    | 0 => r args
    | ix' + 1 => denoteRegionByIx rs' ix' args

-- Morphism from OpM to topM
def OpM.toTopM (rs0: List (TypedArgs Δ → TopM Δ (TypedArgs Δ))):
  OpM Δ (TypedArgs Δ) -> TopM Δ (TypedArgs Δ)
| OpM.Unhandled s => TopM.raiseUB s!"unhandled {s}"
| OpM.Ret r => pure r
| OpM.Error s => TopM.raiseUB s
| OpM.RunRegion ix args k => do
       let ret <- denoteRegionByIx rs0 ix args
       OpM.toTopM rs0 (k ret)

mutual
variable (Δ: Dialect α' σ' ε') [S: Semantics Δ]

-- unfolded version of List.map denoteRegion.
-- This allows the termination checker to view the termination.
def mapDenoteRegion:
  List (Region Δ) →
  List (TypedArgs Δ → TopM Δ (TypedArgs Δ))
| [] => []
| r :: rs => (denoteRegion r) :: mapDenoteRegion rs

-- Convert a region to its denotation to establish finiteness.
-- Then use this finiteness condition to evaluate region semantics.
-- Use the morphism from OpM to TopM.
def denoteOp (op: Op Δ):
    TopM Δ (TypedArgs Δ) :=
  match op with
  | .mk name res0 args0 regions0 attrs => do
      let regionSemantics := mapDenoteRegion regions0
      let resTy := res0.map Prod.snd
      let args ← args0.mapM (fun (name, τ) => do
        pure ⟨τ, ← TopM.get τ name⟩)
      -- Built the interpreted operation
      let iop : IOp Δ := IOp.mk name resTy args (denoteRegionsOpM regions0 0) attrs
      -- Use the dialect-provided semantics, and substitute regions
      let ret ← OpM.toTopM regionSemantics (S.semantics_op iop)
      match (res0, ret) with
      | ([res], [⟨τ, v⟩]) =>
          TopM.set τ res.fst v
          return ret
      | ([], []) =>
          return ret
      | _ =>
        TopM.raiseUB s!"more than one result or unmatched result/expected pair: {op}"
-- denote a sequence of ops
def denoteOps (stmts: List (Op Δ)): TopM Δ (TypedArgs Δ) :=
   match stmts with
   | [] => return  [⟨.unit, ()⟩]
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

/-
def denoteRegions (rs: List (Region Δ)):
    List (TypedArgs Δ → Fitree (SSAEnvE Δ +' UBE) (BlockResult Δ)) :=
 match rs with
 | [] => []
 | r :: rs => (denoteRegion r) :: denoteRegions rs

def denoteRegion (r: Region Δ):
    TypedArgs Δ → Fitree (SSAEnvE Δ +'UBE) (BlockResult Δ) :=
  fun args =>
  -- We only define semantics for single-basic-block regions
  -- Furthermore, we tacticly assume that the region that we run will
  -- return a `BlockResult.Ret`, since we don't bother handling
  -- `BlockResult.Branch`.
  match r with
  | .mk [bb] => do
      let result ← denoteBB bb args

  | _ =>
      raiseUB s!"invalid denoteRegion (>1 bb): {r}"
-/

end

namespace Retraction

variable {α₁ σ₁ ε₁} {δ₁: Dialect α₁ σ₁ ε₁}
 variable   {α₂ σ₂ ε₂} {δ₂: Dialect α₂ σ₂ ε₂}

mutual

-- TODO: create holes for things that are unknown? eg. use `undefined?
def MLIRType.retractLeftList: List (MLIRType (δ₁ + δ₂)) -> Option (List (MLIRType δ₁))
| [] => .some []
| t::ts =>  do
   let ts' <- MLIRType.retractLeftList ts
   let t' <- (MLIRType.retractLeft t)
   return t'::ts'

-- TODO: create holes for things that are unknown? eg. use `undefined?
def MLIRType.retractLeft: MLIRType (δ₁ + δ₂) → Option (MLIRType δ₁)
| .fn argty retty => do
    let argty' <- MLIRType.retractLeft argty
    let retty' <- MLIRType.retractLeft retty
      return (.fn argty' retty')
| .int sgn sz => .some (.int sgn sz) -- : Signedness -> Nat -> MLIRType δ
| .float sz => .some (.float sz) -- : Nat -> MLIRType δ
| .index => .some (.index) --:  MLIRType δ
| .tuple ts => do
    let ts' <- MLIRType.retractLeftList ts
    return .tuple ts'
| .undefined s => .some (.undefined s) -- : String → MLIRType δ
| .extended (Sum.inl σ₁) => .some (.extended σ₁) -- : σ → MLIRType δ
| .extended (Sum.inr σ₂) => .none
end

mutual
def MLIRType.swapDialectList: List (MLIRType (δ₁ + δ₂)) -> List (MLIRType (δ₂ + δ₁))
| [] => []
| t::ts =>  (MLIRType.swapDialect t) :: (MLIRType.swapDialectList ts)


def MLIRType.swapDialect: MLIRType (δ₁ + δ₂) -> MLIRType (δ₂ + δ₁)
| .fn argty retty =>
      .fn (MLIRType.swapDialect argty) (MLIRType.swapDialect retty)
| .int sgn sz => (.int sgn sz) -- : Signedness -> Nat -> MLIRType δ
| .float sz => (.float sz) -- : Nat -> MLIRType δ
| .index => (.index) --:  MLIRType δ
| .tuple ts => .tuple (MLIRType.swapDialectList ts)
| .undefined s => (.undefined s) -- : String → MLIRType δ
| .extended (Sum.inl σ₁) => .extended (Sum.inr σ₁)
| .extended (Sum.inr σ₂) => .extended (Sum.inl σ₂)
end

def TypedArg.swapDialect: TypedArg (δ₁ + δ₂) -> TypedArg (δ₂ + δ₁)
| ⟨.fn argty retty, v⟩ =>
    ⟨.fn (MLIRType.swapDialect argty) (MLIRType.swapDialect retty), () ⟩
| ⟨.int sgn sz, v ⟩ =>  ⟨ .int sgn sz, v ⟩ -- : Signedness -> Nat -> MLIRType δ
| ⟨ .float sz, v ⟩ =>  ⟨.float sz, v ⟩ -- : Nat -> MLIRType δ
| ⟨.index, v⟩ => ⟨.index, v ⟩ --:  MLIRType δ
| ⟨.tuple ts, vs ⟩ =>
   -- TODO: need to convert from (vs: MLIRType.eval (.tuple ts)) to List (TypedArg δ)
   sorry
| ⟨.undefined s, v ⟩ =>  ⟨.undefined s, v⟩ -- : String → MLIRType δ
| ⟨.extended (Sum.inl σ₁), v ⟩ => ⟨.extended (Sum.inr σ₁), v⟩
| ⟨.extended (Sum.inr σ₂), v ⟩ => ⟨.extended (Sum.inl σ₂), v⟩




mutual
def TypedArg.retractLeftList: List (TypedArg (δ₁ + δ₂)) -> Option (List (TypedArg δ₁))
| [] => .some []
| t::ts =>  do
   let ts' <- retractLeftList ts
   let t' <- (TypedArg.retractLeft t)
   return t'::ts'


-- TODO: there is no need to have both a tuple type, and a list of
-- typed args. That's just madness.
def TypedArg.retractLeft (t: TypedArg (δ₁ + δ₂)):  Option (TypedArg δ₁) :=
match t with
| ⟨.fn argty retty, v⟩ =>
    match MLIRType.retractLeft argty with
    | .none => .none
    | .some argty' =>
        match MLIRType.retractLeft retty with
        | .none => .none
        | .some retty' =>
           .some ⟨.fn argty' retty', () ⟩
| ⟨.int sgn sz, v ⟩ => .some ⟨ .int sgn sz, v ⟩ -- : Signedness -> Nat -> MLIRType δ
| ⟨ .float sz, v ⟩ => .some ⟨.float sz, v ⟩ -- : Nat -> MLIRType δ
| ⟨.index, v⟩ => .some ⟨.index, v ⟩ --:  MLIRType δ
| ⟨.tuple ts, vs ⟩ =>
   -- TODO: need to convert from (vs: MLIRType.eval (.tuple ts)) to List (TypedArg δ)
   sorry
| ⟨.undefined s, v ⟩ => .some ⟨.undefined s, v⟩ -- : String → MLIRType δ
| ⟨.extended (Sum.inl σ₁), v ⟩ => .some ⟨.extended σ₁, v⟩ -- : σ → MLIRType δ
| ⟨.extended (Sum.inr σ₂), v ⟩ => .none
end

def TypedArgs.retractLeft: TypedArgs (δ₁ + δ₂) -> Option (TypedArgs δ₁)
| [] => .some []
| tv::ts =>
  match TypedArg.retractLeft tv with
  | .none => .none
  | .some tv' =>
       match TypedArgs.retractLeft ts with
       | .some ts' => .some $ tv'::ts'
       | .none => .none

-- TODO: define the attribute dictionary retraction.
-- Will need to rectact over entries, which will need a retraction over values.
def AttrDict.retractLeft: AttrDict (δ₁ + δ₂) -> Option (AttrDict δ₁)
| _ => .some (AttrDict.mk [])

def AttrDict.swapDialect: AttrDict (δ₁ + δ₂) -> AttrDict (δ₂ + δ₁)
| _ => AttrDict.mk []


def IOp.swapDialect: IOp (δ₁ + δ₂) -> IOp (δ₂ + δ₁) := sorry
/-
| IOp.mk  (name:    String) -- TODO: name should come from an Enum in δ.
  (resTy:   List (MLIRType (δ₁ + δ₂)))
  (args:    TypedArgs (δ₁ + δ₂))
  (regions: List (TypedArgs (δ₁ + δ₂) -> OpM (δ₁ + δ₂) (TypedArgs (δ₁ + δ₂))))
  (attrs:   AttrDict (δ₁ + δ₂)) =>
     IOp.mk name
        (resTy.map MLIRType.swapDialect)
        (args.map TypedArg.swapDialect)
        (AttrDict.swapDialect attrs)
        (regions := sorry)
-/

-- Retract an IOp to the left component.
def IOp.retractLeft: IOp (δ₁ + δ₂) -> Option (IOp δ₁) := sorry
/-
| IOp.mk  (name:    String) -- TODO: name should come from an Enum in δ.
  (resTy:   List (MLIRType (δ₁ + δ₂)))
  (args:    TypedArgs (δ₁ + δ₂))
  (regions: List (TypedArgs (δ₁ + δ₂) -> OpM (δ₁+δ₂) (TypedArgs (δ₁ + δ₂))))
  (attrs:   AttrDict (δ₁ + δ₂)) =>
  match MLIRType.retractLeftList resTy with
  | .none => .none
  | .some resTy' =>
    match TypedArg.retractLeftList args with
    | .none => .none
    | .some args' =>
        match AttrDict.retractLeft attrs with
        | .none => .none
        | .some attrs' => .some (IOp.mk name resTy' args' regions attrs')
-/

def IOp.retractRight (op: IOp (δ₁ + δ₂)): Option (IOp δ₂) :=
  IOp.retractLeft (IOp.swapDialect op)

mutual
def TypedArgs.injectLeft (ts: TypedArgs (δ₁)): TypedArgs (δ₁ +  δ₂) := sorry
def TypedArg.injectLeft (ts: TypedArg (δ₁)): TypedArg (δ₁ +  δ₂) := sorry
def TypedArgs.injectRight (ts: TypedArgs (δ₂)): TypedArgs (δ₁ + δ₂) := sorry
def TypedArg.injectRight (ts: TypedArg (δ₂)): TypedArg (δ₁ +  δ₂) := sorry
end
-- need a way to inject args into larger space
def TypedArgs.retractRight: TypedArgs (δ₁ + δ₂) -> Option (TypedArgs δ₂) := sorry

def injectSemanticsRight [Inhabited R]: OpM δ₂ R -> OpM (δ₁ + δ₂) R := sorry
/-
| .Ret r => .Ret r
| .Vis  (.inl regione) k =>
   match regione with  -- need the match to expose that T = BlockResult δ₁
   | .RunRegion ix args =>
     .Vis (.inl (.RunRegion ix (TypedArgs.injectRight args))) (fun t =>
          match BlockResult.retractRight t with
          | .some t' => injectSemanticsRight (k t')
          | .none =>  .Vis (.inr (UBE.Unhandled (α := Unit))) (fun _ =>  default))
| .Vis (.inr ube) k => .Vis (.inr ube) (fun t => injectSemanticsRight (k t))
-/

def injectSemanticsLeft [Inhabited R]: OpM δ₁ R -> OpM (δ₁ + δ₂)  R := sorry
/-
| .Ret r => .Ret r
| .Vis  (.inl regione) k =>
   match regione with  -- need the match to expose that T = BlockResult δ₁
   | .RunRegion ix args =>
     .Vis (.inl (.RunRegion ix (TypedArgs.injectLeft args))) (fun t =>
          match BlockResult.retractLeft t with
          | .some t' => injectSemanticsLeft (k t')
          | .none =>  .Vis (.inr (UBE.Unhandled (α := Unit))) (fun _ =>  default))
| .Vis (.inr ube) k => .Vis (.inr ube) (fun t => injectSemanticsLeft (k t))
-/

end Retraction

open Retraction in
instance
    {α₁ σ₁ ε₁} {δ₁: Dialect α₁ σ₁ ε₁}
    {α₂ σ₂ ε₂} {δ₂: Dialect α₂ σ₂ ε₂}
    [S₁: Semantics δ₁]
    [S₂: Semantics δ₂]
    : Semantics (δ₁ + δ₂) where
  -- semantics_op: IOp Δ → Fitree (RegionE Δ +' UBE) (BlockResult Δ)
  semantics_op op :=
    -- TODO: need injections of RegionE δ₁ -> RegionE (δ₁ + δ₂)
    -- TODO: need injection of BlockResult δ₁ -> BlockResult (δ₁ + δ₂)
    -- TODO: Can I run both, and somehow interleave the two?
    match Retraction.IOp.retractLeft  op with
    | .some op₁ =>
            (injectSemanticsLeft (S₁.semantics_op op₁)).map TypedArgs.injectLeft
    | .none =>
        match Retraction.IOp.retractRight op with
        | .some op₂ => (injectSemanticsRight (S₂.semantics_op op₂)).map TypedArgs.injectRight
        | .none => OpM.Error  "unknown mixture of dialects"



def run! {Δ: Dialect α' σ' ε'}  {R} [Inhabited R]
    (t: TopM Δ R) (env: SSAEnv Δ):
    R × SSAEnv Δ :=
   match t.run env with
   | .error err => panic! s!"error when running progam: {err}"
   | .ok val => val

def run {Δ: Dialect α' σ' ε'} {R}
    (t: TopM  Δ R) (env: SSAEnv Δ):
    Except String (R × SSAEnv Δ) :=
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
