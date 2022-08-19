/-
## Dialect semantics

This file defines the interface that dialects provide to define their
semantics. This is built upon the `Dialect` interface from `MLIR.Dialects`
which define the custom attributes and type required to model the programs.
-/

import MLIR.Semantics.Fitree
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.AST
open MLIR.AST


-- | Abbreviation with a typeclass context?
@[simp]
abbrev TypedArgs (δ: Dialect α σ ε) := List ((τ: MLIRType δ) × MLIRType.eval τ)

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
| Branch (bb: BBName) (args: List SSAVal)
| Ret (rets:  TypedArgs δ)
| Next (val: (τ: MLIRType δ) × τ.eval)

def BlockResult.toTypedArgs {δ: Dialect α σ ε} (blockResult: BlockResult δ) :=
  match blockResult with
  | .Branch bb args => []
  | .Ret rets => rets
  | .Next val => []

instance : Inhabited (BlockResult δ) where
  default := .Ret []

instance (δ: Dialect α σ ε): ToString (BlockResult δ) where
  toString := fun
    | .Branch bb args => s!"Branch {bb} {args}"
    | .Ret rets       => s!"Ret {rets}"
    | .Next ⟨τ, val⟩  => s!"Next {val}: {τ}"

-- Interpreted operation, like MLIR.AST.Op, but with less syntax
inductive IOp (δ: Dialect α σ ε) := | mk
  (name:    String)
  (resTy:   List (MLIRType δ))
  (args:    TypedArgs δ)
  (bbargs:  List BBName)
  (regions: Nat)
  (attrs:   AttrDict δ)

-- Effect to run a region
-- TODO: change this to also deal with scf.if and yield.
inductive RegionE (Δ: Dialect α σ ε): Type -> Type
| RunRegion (ix: Nat) (args: TypedArgs Δ): RegionE Δ (BlockResult Δ)

class Semantics (δ: Dialect α σ ε)  where
  -- Events modeling the dialect's computational behavior. Usually operations
  -- are simply denoted by a trigger of such an event. This excludes, however,
  -- operations that have regions or otherwise call into other dialects, since
  -- every operation in the program must be accessible before we start
  -- interpreting.
  E: Type → Type

  -- Operation semantics function: maps an IOp (morally an Op but slightly less
  -- rich to guarantee good properties) to an interaction tree. Usually exposes
  -- any region calls then emits of event of E. This function runs on the
  -- program's entire dialect Δ but returns none for any operation that is not
  -- part of δ.
  semantics_op:
    IOp Δ →
    Option (Fitree (RegionE Δ +' UBE +' E) (BlockResult Δ))

  -- TODO: Allow a dialects' semantics to specify their terminators along with
  -- TODO| their branching behavior, instead of hardcoding it for cf

  -- Event handler used when interpreting the operations and running programs.
  -- This is where most of the computational semantics take place.
  -- TODO: Allow dialect handlers to emit events into other dialects
  handle: E ~> Fitree Void1

-- This attribute allows matching explicit effect families like `ArithE`, which
-- often appear from bottom-up inference like in `Fitree.trigger`, with their
-- implicit form like `Semantics.E arith`, which often appears from top-down
-- type inference due to the type signatures in this module. Without this
-- attribute, instances of `Member` could not be derived, preventing most
-- lemmas about `Fitree.trigger` from applying.
attribute [reducible] Semantics.E

mutual
variable (Δ: Dialect α' σ' ε') [S: Semantics Δ]

def interpRegion
    (regions: List <|
      TypedArgs Δ → Fitree (SSAEnvE Δ +' S.E +' UBE) (BlockResult Δ)):
    RegionE Δ +' UBE +' Semantics.E Δ ~>
    Fitree (SSAEnvE Δ +' Semantics.E Δ +' UBE) := fun _ e =>
  match e with
  | Sum.inl (RegionE.RunRegion i xs) => regions.get! i xs
  | Sum.inr <| Sum.inl ube => Fitree.trigger ube
  | Sum.inr <| Sum.inr se => Fitree.trigger se

def denoteOpBase (op: Op Δ):
    Fitree (SSAEnvE Δ +' S.E +' UBE) (BlockResult Δ) :=
  match op with
  | .mk name res0 args0 bbargs regions0 attrs => do
      -- Read arguments from memory
      let args ← args0.mapM (fun (name, τ) => do
          return ⟨τ, ← Fitree.trigger <| SSAEnvE.Get τ name⟩)
      -- Evaluate regions
      -- We write it this way to make the structurral recursion
      -- clear to lean.
      let regions := denoteRegions regions0
      -- Get the result types
      let resTy := res0.map Prod.snd
      -- Built the interpreted operation
      let iop : IOp Δ := IOp.mk name resTy args bbargs regions0.length attrs
      -- Use the dialect-provided semantics, and substitute regions
      match S.semantics_op iop with
      | some t =>
          t.interp (interpRegion regions)
      | none =>
          raiseUB s!"invalid op: {op}"

def denoteOp (op: Op Δ):
    Fitree (SSAEnvE Δ +' S.E +' UBE) (BlockResult Δ) :=
  match op with
  | .mk name [] args0 bbargs regions0 attrs => do
      denoteOpBase op
  | .mk name [(res, _)] args0 bbargs regions0 attrs => do
      let br ← denoteOpBase op
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
  : Fitree (SSAEnvE Δ +' S.E +' UBE) (BlockResult Δ) :=
 match stmts with
 | [] => return BlockResult.Next ⟨.unit, ()⟩
 | [stmt] => denoteOp stmt
 | stmt::stmts => do
      let _ ← denoteOp stmt
      denoteOps stmts

def denoteBB (bb: BasicBlock Δ) (args: TypedArgs Δ):
    Fitree (SSAEnvE Δ +' S.E +' UBE) (BlockResult Δ) := do
  match bb with
  | BasicBlock.mk name formalArgsAndTypes ops =>
     -- TODO: check that types in [TypedArgs] is equal to types at [bb.args]
     -- TODO: Any checks on the BlockResults of intermediate ops?
     let formalArgs : List SSAVal := formalArgsAndTypes.map Prod.fst
     denoteTypedArgs args formalArgs
     denoteOps ops

def denoteRegions (rs: List (Region Δ)):
    List (TypedArgs Δ → Fitree (SSAEnvE Δ +' S.E +' UBE) (BlockResult Δ)) :=
 match rs with
 | [] => []
 | r :: rs => (denoteRegion r) :: denoteRegions rs

def denoteRegion (r: Region Δ):
    TypedArgs Δ → Fitree (SSAEnvE Δ +' S.E +' UBE) (BlockResult Δ) :=
  fun args =>
  -- We only define semantics for single-basic-block regions
  -- Furthermore, we tacticly assume that the region that we run will
  -- return a `BlockResult.Ret`, since we don't bother handling
  -- `BlockResult.Branch`.
  match r with
  | .mk [bb] =>
      denoteBB bb args
  | _ =>
      raiseUB s!"invalid denoteRegion (>1 bb): {r}"
end

instance
    {α₁ σ₁ ε₁} {δ₁: Dialect α₁ σ₁ ε₁}
    {α₂ σ₂ ε₂} {δ₂: Dialect α₂ σ₂ ε₂}
    [S₁: Semantics δ₁]
    [S₂: Semantics δ₂]
    : Semantics (δ₁ + δ₂) where
  E := S₁.E +' S₂.E
  semantics_op op :=
    (S₁.semantics_op op).map (.translate Member.inject) <|>
    (S₂.semantics_op op).map (.translate Member.inject)
  handle := Fitree.case S₁.handle S₂.handle



def semanticsRegionRec
    [S: Semantics Δ]
    (fuel: Nat) (r: Region Δ) (bb: BasicBlock Δ) (entryArgs: TypedArgs Δ):
    Fitree (SSAEnvE Δ +' S.E +' UBE) (BlockResult Δ) :=
  match fuel with
  | 0 => return .Next ⟨.unit, ()⟩
  | fuel' + 1 => do
      match ← denoteBB Δ bb entryArgs with
        | .Branch bbname args =>
            match r.getBasicBlock bbname with
            | some bb' => semanticsRegionRec fuel' r bb' []
            | none => return .Next ⟨.unit, ()⟩
        | .Ret rets => return .Ret rets
        | .Next v => return .Next v

-- TODO: Pass region arguments
-- TODO: Forward region's return type and value
def semanticsRegion {Δ: Dialect α σ ε} [S: Semantics Δ]
    (fuel: Nat) (r: Region Δ) (entryArgs: TypedArgs Δ):
    Fitree (SSAEnvE Δ +' S.E +' UBE) Unit := do
  let _ ← semanticsRegionRec fuel r (r.bbs.get! 0) entryArgs


def run! {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {R}
    (t: Fitree (SSAEnvE Δ +' S.E +' UBE) R) (env: SSAEnv Δ):
    R × SSAEnv Δ :=
  let t := interpSSA' t env
  let t := t.interp' S.handle
  let t := interpUB! t
  t.run

def run {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {R}
    (t: Fitree (SSAEnvE Δ +' S.E +' UBE) R) (env: SSAEnv Δ):
    Except String (R × SSAEnv Δ) :=
  let t := interpSSA' t env
  let t := t.interp' S.handle
  let t := interpUB t
  Fitree.run t

def runLogged {Δ: Dialect α' σ' ε'} [S: Semantics Δ] {R}
    (t: Fitree (SSAEnvE Δ +' S.E +' UBE) R) (env: SSAEnv Δ):
    Except String ((R × String) × SSAEnv Δ) :=
  let t := (interpSSALogged' t).run env
  let t := t.interp' S.handle
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
  denote: T δ → Fitree (SSAEnvE δ +' S.E +' UBE) (BlockResult δ)

notation "⟦ " t " ⟧" => Denote.denote t

instance DenoteOp (δ: Dialect α σ ε) [Semantics δ]: Denote δ Op where
  denote op := denoteOp δ op

-- TODO: this a small hack, because we assume the basic block has no arguments
instance DenoteBB (δ: Dialect α σ ε) [Semantics δ]: Denote δ BasicBlock where
  denote bb := denoteBB δ bb []

-- This only works for single-BB regions with no arguments
instance DenoteRegion (δ: Dialect α σ ε) [Semantics δ]: Denote δ Region where
  denote r := denoteRegion δ r []

-- Not for regions because we need to specify the fuel

@[simp] theorem Denote.denoteOp [Semantics δ]:
  Denote.denote (self := DenoteOp δ) op = denoteOp δ op := rfl
@[simp] theorem Denote.denoteBB [Semantics δ]:
  Denote.denote (self := DenoteBB δ) bb = denoteBB δ bb [] := rfl
@[simp] theorem Denote.denoteRegion [Semantics δ]:
  Denote.denote (self := DenoteRegion δ) r = denoteRegion δ r [] := rfl
