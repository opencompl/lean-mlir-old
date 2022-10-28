/-
## Rewriting of MLIR programs
This file implements a rewriting system for MLIR, following the ideas of PDL.
-/

import MLIR.AST
import MLIR.Semantics.Matching
import MLIR.Semantics.Semantics
import MLIR.Semantics.Refinement
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
-/
structure PeepholeRewriteOp (δ: Dialect α σ ε) [S: Semantics δ] where
  findRoot: MTerm δ
  findSubtree: List (MTerm δ)
  replaceSubtree: List (MTerm δ)
  correct:
    ∀ (findProg: Op δ)
      (replacedProg: List (Op δ))
      (ctx: VarCtx δ)
      (MATCH: matchMProgInOp findProg (findSubtree ++ [findRoot]) [] = .some (_, ctx))
      (FIND: MTerm.concretizeProg (findSubtree ++ [findRoot]) ctx = .some foundProg)
      (SUBST: MTerm.concretizeProg replaceSubtree ctx = .some replacedProg)
      ,  (denoteOps (Δ := δ) (S := S) replacedProg).refines (denoteOps (Δ := δ) (S := S) foundProg).run
