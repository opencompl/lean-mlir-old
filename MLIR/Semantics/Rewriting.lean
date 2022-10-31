/-
## Rewriting of MLIR programs
This file implements a rewriting system for MLIR, following the ideas of PDL.
-/

import MLIR.AST
import MLIR.Semantics.Matching
import MLIR.Semantics.Semantics
import MLIR.Semantics.Refinement
import MLIR.Semantics.Dominance
open MLIR.AST


/-
### replace an operation with multiple operations
The operation to replace is identified by the name of its only result.

TODO: Remove this restriction, and have a way to identify uniquely operations.
-/

mutual
variable (nameMatch: SSAVal) (new_ops: List (Op δ))

def replaceOpInOp (op: Op δ) : Op δ :=
  match op with
  | .mk name res args regions attrs =>
    Op.mk name res args (replaceOpInRegions regions) attrs

def replaceOpInRegions (regions: List (Region δ)) : List (Region δ) :=
  match regions with
  | [] => []
  | region::regions' =>
    (replaceOpInRegion region)::(replaceOpInRegions regions')

def replaceOpInRegion (region: Region δ) : Region δ :=
  match region with
  | .mk name args ops => Region.mk name args (replaceOpInOps ops)

def replaceOpInOps (stmts: List (Op δ)) : List (Op δ) :=
  match stmts with
  | [] => []
  | op::ops' =>
    (replaceOpInOp op)::(replaceOpInOps ops')
end

/-
A peephole rewrite for operations.

x = ?y + ?y -| findSubtree
z = x + x -|
w = z + z -- findRoot


=replace=>

a = (?p - ?p + 1) * ?y |
w = a + a | replaceSubtree
-/
structure PeepholeRewriteOp (δ: Dialect α σ ε) [S: Semantics δ] where
  findRoot: MTerm δ
  findSubtree: List (MTerm δ)
  replaceSubtree: List (MTerm δ)
  wellformed:
    ∀ (toplevelProg: Op δ)
      (_prog: List (Op δ))
      (foundProg: List (Op δ))
      (replacedProg: List (Op δ))
      (substitution: VarCtx δ)
      (domctx: DomContext δ)
      (MATCH: matchMProgInOp toplevelProg (findSubtree ++ [findRoot]) [] = .some (_prog, substitution))
      (FIND: MTerm.concretizeProg (findSubtree ++ [findRoot]) substitution = .some foundProg)
      (SUBST: MTerm.concretizeProg replaceSubtree substitution = .some replacedProg)
      (DOMFIND: (singleBBRegionOpsObeySSA foundProg domctx).isSome = true)
      , (singleBBRegionOpsObeySSA replacedProg domctx).isSome = true

  -- correct:
  --   ∀ (toplevelProg: Op δ)
  --     (replacedProg: List (Op δ))
  --     (substitution: VarCtx δ) -- unification results (substitutions)
  --     (domctx: DomContext δ) -- typing + dominance context
  --     (MATCH: matchMProgInOp toplevelProg (findSubtree ++ [findRoot]) [] =
  --       .some (foundProg, substitution))
  --     -- (FIND: MTerm.concretizeProg (findSubtree ++ [findRoot]) substitution = .some foundProg)
  --     (SUBST: MTerm.concretizeProg replaceSubtree substitution = .some replacedProg)
  --     ,  (denoteOps (Δ := δ) (S := S) replacedProg).refines (denoteOps (Δ := δ) (S := S) foundProg).run

  replace_metavars_subset_find_metavars:
  ∀ (substitution: VarCtx δ) -- unification results (substitutions)
    (SUBST: MTerm.concretizeProg (findSubtree ++ [findRoot]) substitution = .some findProg),
    MTerm.concretizeProg replaceSubtree substitution ≠ none

  /-
  Note that this is sufficient to establish correctness.

  -/
  correct':
  ∀ (substitution: VarCtx δ) -- unification results (substitutions)
    (SUBST: MTerm.concretizeProg (findSubtree ++ [findRoot]) substitution = .some findProg)
    (SUBST': MTerm.concretizeProg replaceSubtree substitution = .some replacedProg)
      ,  (denoteOps (Δ := δ) (S := S) replacedProg).refines
        (denoteOps (Δ := δ) (S := S) findProg).run
