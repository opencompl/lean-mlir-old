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

-- | TODO: throw error if we don't have enough names
def denoteTypedArgs [Member (SSAEnvE Δ) E] (args: TypedArgs Δ) (names: List SSAVal): Fitree E Unit :=
 match args with
 | [] => return ()
 | ⟨τ, val⟩::args =>
    match names with
    | [] => return ()
    | name :: names => do
        Fitree.trigger (SSAEnvE.Set τ name val)
        return ()


-- TODO: Consider changing BlockResult.Branch.args into
--       a TypedArgs (?)
inductive BlockResult {α σ ε} (δ: Dialect α σ ε)
| Ret (rets:  TypedArgs δ)
| Next (val: TypedArg δ) -- Waht the hell is next? I no longer recall...

def BlockResult.toTypedArgs {δ: Dialect α σ ε} (blockResult: BlockResult δ) :=
  match blockResult with
  | .Ret rets => rets
  | .Next val => []

instance : Inhabited (BlockResult δ) where
  default := .Ret []

instance (δ: Dialect α σ ε): ToString (BlockResult δ) where
  toString := fun
    | .Ret rets       => s!"Ret {rets}"
    | .Next ⟨τ, val⟩  => s!"Next {val}: {τ}"

-- Interpreted operation, like MLIR.AST.Op, but with less syntax
inductive IOp (δ: Dialect α σ ε) := | mk
  (name:    String) -- TODO: name should come from an Enum in δ.
  (resTy:   List (MLIRType δ))
  (args:    TypedArgs δ)
  (regions: Nat)
  (attrs:   AttrDict δ)

-- Effect to run a region
-- TODO: change this to also deal with scf.if and yield.
inductive RegionE (Δ: Dialect α σ ε): Type -> Type
-- | TODO: figure out how to coerce BlockResult properly.
| RunRegion (ix: Nat) (args: TypedArgs Δ): RegionE Δ (BlockResult Δ)



class Semantics (Δ: Dialect α σ ε)  where
  -- Operation semantics function: maps an IOp (morally an Op but slightly less
  -- rich to guarantee good properties) to an interaction tree. Usually exposes
  -- any region calls then emits of event of E. This function runs on the
  -- program's entire dialect Δ but returns none for any operation that is not
  -- part of δ.
  semantics_op: IOp Δ → Fitree (RegionE Δ +' UBE) (BlockResult Δ)

-- This attribute allows matching explicit effect families like `ArithE`, which
-- often appear from bottom-up inference like in `Fitree.trigger`, with their
-- implicit form like `Semantics.E arith`, which often appears from top-down
-- type inference due to the type signatures in this module. Without this
-- attribute, instances of `Member` could not be derived, preventing most
-- lemmas about `Fitree.trigger` from applying.
-- attribute [reducible] Semantics.E

mutual
variable (Δ: Dialect α' σ' ε') [S: Semantics Δ]

def denoteOpRegion (regions0: List (Region Δ)) (args: TypedArgs Δ) (ix: Nat): Fitree (SSAEnvE Δ +' UBE) (BlockResult Δ) :=
      match regions0 with
      | [] => raiseUB s!"invalid denoteRegion"
      | r::rs =>
         match ix with
         | 0 =>   denoteRegion r args
         | ix' + 1 => denoteOpRegion rs args ix'

def denoteOpBase
   (name: String)
   (res0: List (TypedSSAVal Δ))
   (args0: List (TypedSSAVal Δ))
   (regions0: List (Region Δ)) (attrs: AttrDict Δ):
    Fitree (SSAEnvE Δ +' UBE) (BlockResult Δ) := do
      -- Read arguments from memory
      let args ← args0.mapM (fun (name, τ) => do
          return ⟨τ, ← Fitree.trigger <| SSAEnvE.Get τ name⟩)
      -- Evaluate regions
      -- Get the result types
      let resTy := res0.map Prod.snd
      -- Built the interpreted operation
      let iop : IOp Δ := IOp.mk name resTy args regions0.length attrs
      -- Use the dialect-provided semantics, and substitute regions
      let t := S.semantics_op iop
      t.interp $ fun K e =>
        match e with
        | Sum.inl (RegionE.RunRegion i args) =>
              denoteOpRegion regions0 args  i
        | Sum.inr ube => Fitree.trigger ube

def denoteOp (op: Op Δ):
    Fitree (SSAEnvE Δ +' UBE) (BlockResult Δ) :=
  match op with
  | .mk name [] args0 regions0 attrs => do
      denoteOpBase name [] args0 regions0 attrs
  | .mk name [(res, resty)] args0 regions0 attrs => do
      let br ← denoteOpBase name [(res, resty)] args0 regions0 attrs
      match br with
      | .Next ⟨τ, v⟩ =>
          -- Should we check that τ is res type here?
          Fitree.trigger (SSAEnvE.Set τ res v)
          return br
      -- TODO: Semi-hack for yields from subregions
      | .Ret [⟨τ, v⟩] =>
          Fitree.trigger (SSAEnvE.Set τ res v)
          return .Next ⟨τ, v⟩
      | _ =>
          return br
  | _ =>
      raiseUB s!"op with more than one result: {op}"

def denoteOps (stmts: List (Op Δ))
  : Fitree (SSAEnvE Δ +' UBE) (BlockResult Δ) :=
 match stmts with
 | [] => return BlockResult.Next ⟨.unit, ()⟩
 | [stmt] => denoteOp stmt
 | stmt::stmts => do
      let _ ← denoteOp stmt
      denoteOps stmts

def denoteRegion (rgn: Region Δ) (args: TypedArgs Δ):
    Fitree (SSAEnvE Δ +' UBE) (BlockResult Δ) := do
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


def IOp.swapDialect: IOp (δ₁ + δ₂) -> IOp (δ₂ + δ₁)
| IOp.mk  (name:    String) -- TODO: name should come from an Enum in δ.
  (resTy:   List (MLIRType (δ₁ + δ₂)))
  (args:    TypedArgs (δ₁ + δ₂))
  (regions: Nat)
  (attrs:   AttrDict (δ₁ + δ₂)) =>
     IOp.mk name
        (resTy.map MLIRType.swapDialect)
        (args.map TypedArg.swapDialect)
        regions
        (AttrDict.swapDialect attrs)

-- Retract an IOp to the left component.
def IOp.retractLeft: IOp (δ₁ + δ₂) -> Option (IOp δ₁)
| IOp.mk  (name:    String) -- TODO: name should come from an Enum in δ.
  (resTy:   List (MLIRType (δ₁ + δ₂)))
  (args:    TypedArgs (δ₁ + δ₂))
  (regions: Nat)
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


def IOp.retractRight (op: IOp (δ₁ + δ₂)): Option (IOp δ₂) :=
  IOp.retractLeft (IOp.swapDialect op)

mutual
def TypedArgs.injectLeft (ts: TypedArgs (δ₁)): TypedArgs (δ₁ +  δ₂) := sorry
def TypedArg.injectLeft (ts: TypedArg (δ₁)): TypedArg (δ₁ +  δ₂) := sorry
def TypedArgs.injectRight (ts: TypedArgs (δ₂)): TypedArgs (δ₁ + δ₂) := sorry
def TypedArg.injectRight (ts: TypedArg (δ₂)): TypedArg (δ₁ +  δ₂) := sorry
end
-- need a way to inject args into larger space
def RegionE.injectLeft: RegionE δ₁ (BlockResult δ₁) -> RegionE (δ₁ + δ₂) (BlockResult (δ₁ + δ₂))
| .RunRegion ix args => .RunRegion ix (TypedArgs.injectLeft args)

def RegionE.injectRight: RegionE δ₂ (BlockResult δ₂) -> RegionE (δ₁ + δ₂) (BlockResult (δ₁ + δ₂))
| .RunRegion ix args => .RunRegion ix (TypedArgs.injectRight args)



def BlockResult.injectLeft: BlockResult δ₁→ BlockResult (δ₁ + δ₂)
| .Ret (rets:  TypedArgs δ₁) => .Ret (TypedArgs.injectLeft rets)
| .Next val => .Next (TypedArg.injectLeft val)

def BlockResult.injectRight: BlockResult δ₂→ BlockResult (δ₁ + δ₂)
| .Ret (rets:  TypedArgs δ₂) => .Ret (TypedArgs.injectRight rets)
| .Next val => .Next (TypedArg.injectRight val)

def BlockResult.retractLeft: BlockResult (δ₁ + δ₂) -> Option (BlockResult δ₁) := sorry
def BlockResult.retractRight: BlockResult (δ₁ + δ₂) -> Option (BlockResult δ₂) := sorry

def injectSemanticsRight [Inhabited R]: Fitree (RegionE δ₂ +' UBE) R
  -> Fitree (RegionE (δ₁ + δ₂) +' UBE) R
| .Ret r => .Ret r
| .Vis  (.inl regione) k =>
   match regione with  -- need the match to expose that T = BlockResult δ₁
   | .RunRegion ix args =>
     .Vis (.inl (.RunRegion ix (TypedArgs.injectRight args))) (fun t =>
          match BlockResult.retractRight t with
          | .some t' => injectSemanticsRight (k t')
          | .none =>  .Vis (.inr (UBE.Unhandled (α := Unit))) (fun _ =>  default))
| .Vis (.inr ube) k => .Vis (.inr ube) (fun t => injectSemanticsRight (k t))

def injectSemanticsLeft [Inhabited R]: Fitree (RegionE δ₁ +' UBE) R
  -> Fitree (RegionE (δ₁ + δ₂) +' UBE) R
| .Ret r => .Ret r
| .Vis  (.inl regione) k =>
   match regione with  -- need the match to expose that T = BlockResult δ₁
   | .RunRegion ix args =>
     .Vis (.inl (.RunRegion ix (TypedArgs.injectLeft args))) (fun t =>
          match BlockResult.retractLeft t with
          | .some t' => injectSemanticsLeft (k t')
          | .none =>  .Vis (.inr (UBE.Unhandled (α := Unit))) (fun _ =>  default))
| .Vis (.inr ube) k => .Vis (.inr ube) (fun t => injectSemanticsLeft (k t))

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
            (injectSemanticsLeft (S₁.semantics_op op₁)).map BlockResult.injectLeft
    | .none =>
        match Retraction.IOp.retractRight op with
        | .some op₂ =>(injectSemanticsRight (S₂.semantics_op op₂)).map BlockResult.injectRight
        | .none => Fitree.trigger $ UBE.UB "unknown mixture of dialects"



def run! {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {R}
    (t: Fitree (SSAEnvE Δ +' UBE) R) (env: SSAEnv Δ):
    R × SSAEnv Δ :=
  let t := interpSSA' t env
  let t := interpUB! t
  t.run

def run {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {R}
    (t: Fitree (SSAEnvE Δ +' UBE) R) (env: SSAEnv Δ):
    Except String (R × SSAEnv Δ) :=
  let t := interpSSA' t env
  let t := interpUB t
  Fitree.run t

def runLogged {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {R}
    (t: Fitree (SSAEnvE Δ +' UBE) R) (env: SSAEnv Δ):
    Except String ((R × String) × SSAEnv Δ) :=
  let t := (interpSSALogged' t).run env
  let t := interpUB t
  Fitree.run t

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
  denote: T δ → Fitree (SSAEnvE δ +'UBE) (BlockResult δ)

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
