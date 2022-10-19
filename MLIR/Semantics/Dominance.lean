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
abbrev DomContext (type: Type) := List (SSAVal × MLIRType type)

-- Add a typed SSA value in the context
def DomContext.addVal (ctx: DomContext type) (val: SSAVal) (τ: MLIRType type) :
    DomContext type :=
  (val, τ)::ctx

-- Return true if an SSA name is already defined
def DomContext.isValDefined (ctx: DomContext type) (val: SSAVal) : Bool :=
  (ctx.find? (val == ·.fst)).isSome

-- Return true if an SSA value has already been defined with the correct type
def DomContext.isValUseCorrect [Code type] (ctx: DomContext type) (val: SSAVal)
    (τ: MLIRType type) : Bool :=
  match (ctx.find? (val == ·.fst)) with
  | some (_, τ') => τ == τ'
  | none => false

-- Check that an SSA value definition is correct, and append it to the context
def valDefinitionObeySSA (val: TypedSSAVal δ) (ctx: DomContext δ)
    : Option (DomContext δ) :=
  if ctx.isValDefined val.fst then none else ctx.addVal val.fst val.snd

-- Check that operands are already defined, with
def operandsDefinitionObeySSA [ct: Code type] (args: List (TypedSSAVal type)) (ctx: DomContext type) : Bool :=
  args.all (λ ⟨val, τ⟩ => ctx.isValUseCorrect val τ)

/-
### Dominance check

Check that an IR satisfies SSA.
-/

mutual
def singleBBRegionOpObeySSA [Code type] (op: Op attr type) (ctx: DomContext type) : Option (DomContext type) :=
  match op with
  | Op.mk _ results operands regions _ => do
    -- Check operands
    let _ ← match operandsDefinitionObeySSA operands ctx with
            | true => pure ctx
            | false => none
    -- Check regions
    let _ <- singleBBRegionRegionsObeySSA regions ctx
    -- Check results
    let ctx' <- match results with
             | [] => .some ctx
             | [result] => valDefinitionObeySSA result ctx
             | _ => none
    ctx'

def singleBBRegionRegionsObeySSA [Code type] (regions: List (Region attr type)) (ctx: DomContext type) : Option (DomContext type) :=
  match regions with
  | region::regions' => do
    let _ <- (singleBBRegionObeySSA region ctx)
    let _ <- (singleBBRegionRegionsObeySSA regions' ctx)
    ctx
  | [] => some ctx

def singleBBRegionObeySSA [Code type] (region: Region attr type) (ctx: DomContext type) : Option (DomContext type) :=
  match region with
  | .mk name args stmts =>
    (args.foldlM (fun ctx arg => valDefinitionObeySSA arg ctx) ctx).bind
    (singleBBRegionOpsObeySSA stmts)


def singleBBRegionOpsObeySSA [Code type](ops: List (Op attr type)) (ctx: DomContext type) : Option (DomContext type) :=
  match ops with
  | op::ops' => (singleBBRegionOpObeySSA op ctx).bind (singleBBRegionOpsObeySSA ops')
  | [] => some ctx
end

/-
termination_by
  singleBBRegionOpObeySSA  op _ => sizeOf op
  singleBBRegionRegionsObeySSA regions _=> sizeOf regions
  singleBBRegionObeySSA rgn _ => sizeOf rgn
  singleBBRegionOpsObeySSA ops _ => sizeOf ops
-/

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
def hasUniqueNamesOp (op: Op attr type) (ctx: NameContext) : Option NameContext :=
  match op with
  | Op.mk _ _ _ regions _ => hasUniqueNamesRegions regions ctx

def hasUniqueNamesRegions (regions: List (Region attr type)) (ctx: NameContext) :
    Option NameContext :=
  match regions with
  | region::regions' => do
    let ctx' <- (hasUniqueNamesRegion region ctx)
    (hasUniqueNamesRegions regions' ctx')
  | [] => none

def hasUniqueNamesRegion (rgn: Region attr type) (ctx: NameContext) :
    Option NameContext :=
  match rgn with
  | .mk _ args ops => do
    let ctx' ←
      args.foldlM (fun ctx arg => valDefHasUniqueNames ctx arg.fst) ctx
    hasUniqueNamesOps ops ctx'

def hasUniqueNamesOps (ops: List (Op attr type)) (ctx: NameContext) :
    Option NameContext :=
  match ops with
  | op::ops' => do
    let ctx' ← hasUniqueNamesOp op ctx
    hasUniqueNamesOps ops' ctx'
  | [] => none
end

/-
### Use-def chain operations

Get the definition of a variable, or check if it is used
-/


mutual
variable (mVar: SSAVal)

def isSSADefInOp (op: Op attr type) : Bool :=
  match op with
  | .mk _ _ _ regions _ => isSSADefInRegions regions

def isSSADefInRegions (regions: List (Region attr type)) : Bool :=
  match regions with
  | [] => False
  | region::regions' => isSSADefInRegion region || isSSADefInRegions regions'

def isSSADefInRegion (rgn: Region attr type) : Bool :=
  match rgn with
  | .mk _ _ ops => isSSADefInOps ops

def isSSADefInOps (ops: List (Op attr type)) : Bool :=
  match ops with
  | [] => False
  | op::ops' => isSSADefInOp op || isSSADefInOps ops'
end


/-
Check if the variable used by the operation.
Do not check inside the regions inside the operation.
-/
def isUsed (var: SSAVal) (op: Op attr type) : Bool :=
  var ∈ op.argNames

/-
Check if `op` is used by `user`.
An operation is used by another operation if one of its
argument is used by the operation.
-/
def isOpUsed (op user: Op attr type) : Bool :=
  op.resNames.any (fun arg => isUsed arg user)


mutual
variable (mVar: SSAVal)

def isSSAUsedInOp (op: Op attr type) : Bool :=
  match op with
  | .mk _ _ _ regions _ =>
    isUsed mVar op || isSSAUsedInRegions regions

def isSSAUsedInRegions (regions: List (Region attr type)) : Bool :=
  match regions with
  | [] => False
  | region::regions' => isSSAUsedInRegion region || isSSAUsedInRegions regions'


def isSSAUsedInRegion (rgn: Region attr type) : Bool :=
  match rgn with
  | .mk _ _ ops => isSSAUsedInOps ops

def isSSAUsedInOps (ops: List (Op attr type)) : Bool :=
  match ops with
  | [] => False
  | op::ops' => isSSAUsedInOp op || isSSAUsedInOps ops'
end


mutual
variable (mVar: SSAVal)

def getDefiningOpInOp (op: Op attr type) : Option (Op attr type) :=
  if mVar ∈ op.resNames then
    some op
  else
    match op with
    | .mk _ _ _ regions _ => getDefiningOpInRegions regions

def getDefiningOpInRegions (regions: List (Region attr type)) : Option (Op attr type) :=
  match regions with
  | [] => none
  | region::regions' =>
    (getDefiningOpInRegion region).orElse
    (fun () => getDefiningOpInRegions regions')

def getDefiningOpInRegion (rgn: Region attr type) : Option (Op attr type) :=
  match rgn with
  | .mk _ _ ops => getDefiningOpInOps ops

def getDefiningOpInOps (ops: List (Op attr type)) : Option (Op attr type) :=
  match ops with
  | [] => none
  | op::ops' =>
    match getDefiningOpInOp op with
    | some op => some op
    | none => getDefiningOpInOps ops'
end
