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
def valDefinitionObeySSA (val: SSAVal) (τ: MLIRType δ) 
                         (ctx: DomContext δ) : Option (DomContext δ) :=
  if ctx.isValDefined val then none else ctx.addVal val τ

-- Check that operands are already defined, with
def operandsDefinitionObeySSA (vals: List SSAVal) (τs: List (MLIRType δ))
                           (ctx: DomContext δ) : Bool :=
  match vals, τs with
  | val::vals', τ::τs' =>
    ctx.isValUseCorrect val τ && operandsDefinitionObeySSA vals' τs' ctx
  | [], [] => true
  | _, _ => false

/-
### Dominance check

Check that an IR satisfies SSA.
-/

mutual
def singleBBRegionOpObeySSA (op: Op δ) (ctx: DomContext δ) : Option (DomContext δ) :=
  match op with
  | Op.mk _ operands [] regions _ (MLIRType.fn (MLIRType.tuple operandsTy) _) => do
    let b := operandsDefinitionObeySSA operands operandsTy ctx
    match b with 
    | true =>  (singleBBRegionRegionsObeySSA regions ctx)
    | false => none
  | _ => none

def singleBBRegionRegionsObeySSA (regions: List (Region δ)) (ctx: DomContext δ) : Option (DomContext δ) :=
  match regions with
  | region::regions' => do 
    let _ <- (singleBBRegionRegionObeySSA region ctx)
    (singleBBRegionRegionsObeySSA regions' ctx)
  | [] => none

def singleBBRegionRegionObeySSA (region: Region δ) (ctx: DomContext δ) : Option (DomContext δ) :=
  match region with
  | .mk [] => ctx
  | .mk [bb] => (singleBBRegionBBObeySSA bb ctx)
  | _ => Option.none

def singleBBRegionBBObeySSA (bb: BasicBlock δ) (ctx: DomContext δ) : Option (DomContext δ) :=
  match bb with
  | .mk _ args stmts =>
    (args.foldlM (fun ctx arg => valDefinitionObeySSA arg.fst arg.snd ctx) ctx).bind 
    (singleBBRegionStmtsObeySSA stmts)

def singleBBRegionStmtsObeySSA (stmts: List (BasicBlockStmt δ)) (ctx: DomContext δ) : Option (DomContext δ) :=
  match stmts with
  | stmt::stmts' => (singleBBRegionStmtObeySSA stmt ctx).bind (singleBBRegionStmtsObeySSA stmts')
  | [] => none

def singleBBRegionStmtObeySSA (stmt: BasicBlockStmt δ) (ctx: DomContext δ) : Option (DomContext δ) :=
  match stmt with
  | .StmtOp op => singleBBRegionOpObeySSA op ctx
  | .StmtAssign res none op => do
    -- TODO: replace it with an `as`, when I'll know how to do it
    let ctx' <- match op with
               | Op.mk _ _ _ _ _ (MLIRType.fn _ (MLIRType.tuple [τ])) => (valDefinitionObeySSA res τ ctx)
               | _ => none
    singleBBRegionOpObeySSA op ctx
  | _ => none
end
termination_by
  singleBBRegionOpObeySSA  op _ => sizeOf op
  singleBBRegionRegionsObeySSA regions _=> sizeOf regions
  singleBBRegionRegionObeySSA region _ => sizeOf region
  singleBBRegionBBObeySSA bb _ => sizeOf bb
  singleBBRegionStmtsObeySSA stmts _ => sizeOf stmts
  singleBBRegionStmtObeySSA stmt _ => sizeOf stmt

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
  | Op.mk _ _ _ regions _ _ => hasUniqueNamesRegions regions ctx

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
  | .mk _ args stmts => do
    let ctx' ←
      args.foldlM (fun ctx arg => valDefHasUniqueNames ctx arg.fst) ctx
    hasUniqueNamesBBStmts stmts ctx'

def hasUniqueNamesBBStmts (stmts: List (BasicBlockStmt δ)) (ctx: NameContext) :
    Option NameContext :=
  match stmts with
  | stmt::stmts' => do
    let ctx' ← hasUniqueNamesBBStmt stmt ctx
    hasUniqueNamesBBStmts stmts' ctx'
  | [] => none

def hasUniqueNamesBBStmt (stmt: BasicBlockStmt δ) (ctx: NameContext) :
    Option NameContext :=
  match stmt with
  | .StmtOp op => hasUniqueNamesOp op ctx
  | .StmtAssign res _ op => do
    let ctx' <- valDefHasUniqueNames ctx res
    hasUniqueNamesOp op ctx'
end
