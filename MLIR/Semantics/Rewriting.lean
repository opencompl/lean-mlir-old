/-
## Rewriting of MLIR programs

This file implements a rewriting system for MLIR, following the ideas of PDL.
-/

import MLIR.Semantics.Matching
import MLIR.Semantics.Semantics
import MLIR.AST
open MLIR.AST

/-
### replace an operation with multiple operations

The operation to replace is identified by the name of its only result.
TODO: Remove this restriction.
-/

mutual
variable (nameMatch: SSAVal) (new_ops: List (BasicBlockStmt δ))

def replaceOpInOp (op: Op δ) : Op δ := 
  match op with
  | .mk name args bbs regions attrs ty => 
    .mk name args bbs (replaceOpInRegions regions) attrs ty

def replaceOpInRegions (regions: List (Region δ)) : List (Region δ) :=
  match regions with
  | [] => []
  | region::regions' => 
    (replaceOpInRegion region)::(replaceOpInRegions regions')

def replaceOpInRegion (region: Region δ) : Region δ :=
  match region with
  | .mk bbs => .mk (replaceOpInBBs bbs)

def replaceOpInBBs (bbs: List (BasicBlock δ)) : List (BasicBlock δ) :=
  match bbs with
  | [] => []
  | bb::bbs' => (replaceOpInBB bb)::(replaceOpInBBs bbs')

def replaceOpInBB (bb: BasicBlock δ) : BasicBlock δ :=
  match bb with
  | .mk name args ops => .mk name args (replaceOpInBBStmts ops)

def replaceOpInBBStmts (stmts: List (BasicBlockStmt δ)) :
    List (BasicBlockStmt δ) :=
  match stmts with
  | [] => []
  | stmt::stmts' =>
    (replaceOpInBBStmt stmt) ++ (replaceOpInBBStmts stmts')

def replaceOpInBBStmt (stmt: BasicBlockStmt δ) : List (BasicBlockStmt δ) :=
  match stmt with
  | .StmtOp op => [.StmtOp (replaceOpInOp op)]
  | .StmtAssign var idx op =>
      if var == nameMatch then
        new_ops
      else 
        [.StmtAssign var idx (replaceOpInOp op)]
end
