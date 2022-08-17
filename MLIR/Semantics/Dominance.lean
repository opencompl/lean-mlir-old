/-
## Dominance check

This files defines functions to check that an IR satisfies SSA.
In particular, this will also check that operations have at most one result,
and each region has at most one block.
-/


import MLIR.AST
import MLIR.Semantics.Types
open MLIR.AST

/-
### Dominance Context

Context that holds the names and types of the SSA values that are
defined in the scope.
-/

-- List of typed values that are in scope
abbrev DomContext (δ: Dialect α σ ε) := List (SSAVal × MLIRType δ)

-- Add a typed SSA value in the context
def DomContext.addVal (ctx: DomContext δ) (val: SSAVal) (τ: MLIRType δ) :
    DomContext δ :=
  (val, τ)::ctx

-- Return true if an SSA name is already defined
def DomContext.isValDefined (ctx: DomContext δ) (val: SSAVal) : Bool :=
  (ctx.find? (val == ·.fst)).isSome

-- Return true if an SSA value has already been defined with the correct type
def DomContext.isValUseCorrect (ctx: DomContext δ) (val: SSAVal)
    (τ: MLIRType δ) : Bool :=
  match (ctx.find? (val == ·.fst)) with
  | some (_, τ') => τ == τ'
  | none => false

-- Check that an SSA value definition is correct, and append it to the context
def valDefinitionObeySSA (val: TypedSSAVal δ) (ctx: DomContext δ)
    : Option (DomContext δ) :=
  if ctx.isValDefined val.fst then none else ctx.addVal val.fst val.snd

-- Check that operands are already defined, with
def operandsDefinitionObeySSA (args: List (TypedSSAVal δ)) (ctx: DomContext δ) : Bool :=
  args.all (λ ⟨val, τ⟩ => ctx.isValUseCorrect val τ)

/-
### Dominance check

Check that an IR satisfies SSA.
-/

mutual
def singleBBRegionOpObeySSA (op: Op δ) (ctx: DomContext δ) : Option (DomContext δ) :=
  match op with
  | Op.mk _ results operands [] regions _ => do
    -- Check operands
    let b := operandsDefinitionObeySSA operands ctx
    -- Check regions
    let _ <- match b with 
             | true => (singleBBRegionRegionsObeySSA regions ctx)
             | false => none
    -- Check results
    let ctx' <- match results with
             | [] => ctx
             | [result] => valDefinitionObeySSA result ctx
             | _ => none
    ctx'
  | _ => none

def singleBBRegionRegionsObeySSA (regions: List (Region δ)) (ctx: DomContext δ) : Option (DomContext δ) :=
  match regions with
  | region::regions' => do 
    let _ <- (singleBBRegionRegionObeySSA region ctx)
    let _ <- (singleBBRegionRegionsObeySSA regions' ctx)
    ctx
  | [] => none

def singleBBRegionRegionObeySSA (region: Region δ) (ctx: DomContext δ) : Option (DomContext δ) :=
  match region with
  | .mk [] => ctx
  | .mk [bb] => (singleBBRegionBBObeySSA bb ctx)
  | _ => Option.none

def singleBBRegionBBObeySSA (bb: BasicBlock δ) (ctx: DomContext δ) : Option (DomContext δ) :=
  match bb with
  | .mk _ args stmts =>
    (args.foldlM (fun ctx arg => valDefinitionObeySSA arg ctx) ctx).bind 
    (singleBBRegionOpsObeySSA stmts)

def singleBBRegionOpsObeySSA (ops: List (Op δ)) (ctx: DomContext δ) : Option (DomContext δ) :=
  match ops with
  | op::ops' => (singleBBRegionOpObeySSA op ctx).bind (singleBBRegionOpsObeySSA ops')
  | [] => none
end
termination_by
  singleBBRegionOpObeySSA  op _ => sizeOf op
  singleBBRegionRegionsObeySSA regions _=> sizeOf regions
  singleBBRegionRegionObeySSA region _ => sizeOf region
  singleBBRegionBBObeySSA bb _ => sizeOf bb
  singleBBRegionOpsObeySSA ops _ => sizeOf ops

/-
### Uniqueness of SSA names

Check that SSA names are unique, even across regions.
This simplifies a lot our proofs.
This is not always implied by Dominance check, since with dominance check,
two regions in a same operation can have operations defining the same ssa name.
-/

-- Contains the names that are already defined
abbrev NameContext := List SSAVal

-- Add a typed SSA value in the context
def NameContext.addVal (ctx: NameContext) (val: SSAVal) : NameContext :=
  val::ctx

-- Return true if an SSA name is already defined
def NameContext.isValDefined (ctx: NameContext) (val: SSAVal) : Bool :=
  (ctx.find? (val == ·)).isSome

-- Check that an SSA value definition has name that wasn't previously defined
def valDefHasUniqueNames (ctx: NameContext) (val: SSAVal)
    : Option NameContext :=
  if ctx.isValDefined val then
    some (ctx.addVal val)
  else
    none

mutual
def hasUniqueNamesOp (op: Op δ) (ctx: NameContext) : Option NameContext :=
  match op with
  | Op.mk _ _ _ _ regions _ => hasUniqueNamesRegions regions ctx

def hasUniqueNamesRegions (regions: List (Region δ)) (ctx: NameContext) :
    Option NameContext :=
  match regions with
  | region::regions' => do 
    let ctx' <- (hasUniqueNamesRegion region ctx)
    (hasUniqueNamesRegions regions' ctx')
  | [] => none

def hasUniqueNamesRegion (region: Region δ) (ctx: NameContext) :
    Option NameContext :=
  match region with
  | .mk bbs => (hasUniqueNamesBBs bbs ctx)

def hasUniqueNamesBBs (bbs: List (BasicBlock δ)) (ctx: NameContext) :
    Option NameContext :=
  match bbs with
  | [] => some ctx
  | bb::bbs' => do
    let ctx' <- (hasUniqueNamesBB bb ctx)
    hasUniqueNamesBBs bbs' ctx'

def hasUniqueNamesBB (bb: BasicBlock δ) (ctx: NameContext) :
    Option NameContext :=
  match bb with
  | .mk _ args ops => do
    let ctx' ←
      args.foldlM (fun ctx arg => valDefHasUniqueNames ctx arg.fst) ctx
    hasUniqueNamesOps ops ctx'

def hasUniqueNamesOps (ops: List (Op δ)) (ctx: NameContext) :
    Option NameContext :=
  match ops with
  | op::ops' => do
    let ctx' ← hasUniqueNamesOp op ctx
    hasUniqueNamesOps ops' ctx'
  | [] => none
end
