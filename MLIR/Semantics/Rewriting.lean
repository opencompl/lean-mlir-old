/-
## Rewriting of MLIR programs

This file implements a rewriting system for MLIR, following the ideas of PDL.
-/

import MLIR.Semantics.Matching
import MLIR.Semantics.Semantics
import MLIR.AST
import MLIR.Semantics.Refinement
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

/-
### MTerm actions

MTerm actions are actions that can be done on a list of op MTerm.
These correspond to PDL rewrites, such as replacing an SSA Value with a new one,
or replacing an operation with multiple operations.
-/

inductive MTermAction (δ: Dialect α σ ε) :=
--| ReplaceValue (oldVal newVal: SSAVal)
| ReplaceOp (varMatch: String) (newOps: List (MTerm δ))

def MTermAction.apply (a: MTermAction δ) (prog: List (BasicBlockStmt δ)) (ctx: VarCtx δ) : Option (List (BasicBlockStmt δ)) :=
  match a with
  | ReplaceOp varMatch newOps => do
    let concreteVarMatch ← ctx.get .MSSAVal varMatch
    let concreteVarMatchName := match concreteVarMatch with | .SSAVal name => name
    let newOps ← newOps.mapM (fun t => t.concretizeOp ctx)
    replaceOpInBBStmts concreteVarMatchName newOps prog

/-
### Simple example

We take the example of a MTerm representing `y = x + x`, that we replace with
`y = x * x`.
-/

private def test_addi_multiple_pattern: List (MTerm δ) :=
  [.App .OP [
    .ConstString "std.addi",
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "op_res" .MSSAVal, .Var 2 "T" .MMLIRType],
      .App .OPERAND [.Var 2 "op_res" .MSSAVal, .Var 2 "T" .MMLIRType]],
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "op_res2" .MSSAVal, .Var 2 "T" .MMLIRType]]
  ]]

private def test_new_ops: List (MTerm builtin) :=
  [.App .OP [
    .ConstString "std.muli",
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "op_res" .MSSAVal, .Var 2 "T" .MMLIRType],
      .App .OPERAND [.Var 2 "op_res" .MSSAVal, .Var 2 "T" .MMLIRType]],
    .App (.LIST .MOperand) [
      .App .OPERAND [.Var 2 "op_res2" .MSSAVal, .Var 2 "T" .MMLIRType]]
  ]]

private def multiple_example: Op builtin := [mlir_op|
  "builtin.module"() ({
    ^entry:
    %r3 = "std.addi"(%r, %r): (i32, i32) -> (i32)
  }) : ()
]

-- Match an MTerm program in some IR, then concretize
-- the MTerm using the resulting matching context.
private def multiple_example_result :
    Option (List (BasicBlockStmt builtin) × List (BasicBlockStmt builtin)) := do
  let (_, ctx) ←
    matchMProgInOp multiple_example test_addi_multiple_pattern []
  let origin_prog ← MTerm.concretizeProg test_addi_multiple_pattern ctx
  let action := MTermAction.ReplaceOp "op_res2" test_new_ops
  let res_prog ← action.apply origin_prog ctx
  (origin_prog, res_prog)

#eval multiple_example_result

/-
### Postcondition of an operation interpretation.

We define the environment postcondition of an operation interpretation by
the set of all possible `SSAEnv` that could result after the interpretation
of the operation.
-/

def postSSAEnv [Semantics δ] (op: BasicBlockStmt δ) (env: SSAEnv δ) : Prop :=
  ∃ env', (run ⟦op⟧ env').snd = env

def postSSAEnvList [Semantics δ] (op: List (BasicBlockStmt δ)) (env: SSAEnv δ) : Prop :=
  ∃ env', (run ⟦BasicBlock.mk "" [] op⟧ env').snd = env


def termResName (m: MTerm δ) : Option String :=
  match m with
  | .App .OP [ _, _, .App (.LIST .MOperand) 
        [.App .OPERAND [.Var _ ssaName .MSSAVal, _]] ] => some ssaName
  | _ => none

def run_split_head_tail [S: Semantics δ] (mHead mTail: List (MTerm δ))
                        (a: MTermAction δ) (σ: VarCtx δ) :
  ∀ pHead, MTerm.concretizeProg mHead σ = some pHead -> 
  ∀ pTail, MTerm.concretizeProg mTail σ = some pTail -> 
  ∀ (env: SSAEnv δ), (run ⟦BasicBlock.mk "" [] (pHead ++ pTail)⟧ env).snd = 
    (run ⟦BasicBlock.mk "" [] pTail⟧ (run ⟦BasicBlock.mk "" [] pHead⟧ env).snd).snd := sorry

def rewrite_equivalent_precondition_rewrite [S: Semantics δ] (mHead: List (MTerm δ))
    (mTail: MTerm δ) (mNewTail: List (MTerm δ)) (σ: VarCtx δ) :
  ∀ headProg, MTerm.concretizeProg mHead σ = some headProg →
  ∀ originProg, MTerm.concretizeProg [mTail] σ = some originProg →
  ∀ resProg, MTerm.concretizeProg mNewTail σ = some resProg →
  (∀ (env: SSAEnv δ), 
    refinement (run ⟦BasicBlock.mk "" [] (headProg ++ originProg)⟧ env)
               (run ⟦BasicBlock.mk "" [] (headProg ++ resProg)⟧ env)) -> 
  ∀ env, postSSAEnvList headProg env ->
    refinement (run ⟦BasicBlock.mk "" [] originProg⟧ env)
               (run ⟦BasicBlock.mk "" [] resProg⟧ env) := by sorry
