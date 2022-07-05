/-
## Rewriting of MLIR programs

This file implements a rewriting system for MLIR, following the ideas of PDL.
-/

import MLIR.AST
import MLIR.Semantics.Matching
import MLIR.Semantics.Semantics
import MLIR.Semantics.Dominance
import MLIR.Semantics.Refinement
open MLIR.AST

/-
### replace an operation with multiple operations

The operation to replace is identified by the name of its only result.
TODO: Remove this restriction.
-/

mutual
variable (nameMatch: SSAVal) (new_ops: List (BasicBlockStmt δ))

def replaceOpInOp (op: Op δ) : Option (Op δ) := 
  match op with
  | .mk name args bbs regions attrs ty => do
    let regions' ← replaceOpInRegions regions
    Op.mk name args bbs regions' attrs ty

def replaceOpInRegions (regions: List (Region δ)) : Option (List (Region δ)) :=
  match regions with
  | [] => none
  | region::regions' =>
    match replaceOpInRegion region with
    | some region' => region'::regions'
    | none => do
        let regions'' ← replaceOpInRegions regions'
        region::regions''

def replaceOpInRegion (region: Region δ) : Option (Region δ) :=
  match region with
  | .mk bbs => do 
    let bbs' ← replaceOpInBBs bbs
    Region.mk bbs'

def replaceOpInBBs (bbs: List (BasicBlock δ)) : Option (List (BasicBlock δ)) :=
  match bbs with
  | [] => none
  | bb::bbs' => 
    match replaceOpInBB bb with
    | some bb' => bb'::bbs'
    | none => do
        let bbs'' ← replaceOpInBBs bbs'
        bb::bbs''

def replaceOpInBB (bb: BasicBlock δ) : Option (BasicBlock δ) :=
  match bb with
  | .mk name args ops => do
      let ops' ← replaceOpInBBStmts ops
      BasicBlock.mk name args ops'

def replaceOpInBBStmts (stmts: List (BasicBlockStmt δ)) :
    Option (List (BasicBlockStmt δ)) :=
  match stmts with
  | [] => none
  | stmt::stmts' =>
    match replaceOpInBBStmt stmt with
    | some stmt' => some (stmt' ++ stmts')
    | none => do
        let stmts'' ← replaceOpInBBStmts stmts'
        stmt::stmts''

def replaceOpInBBStmt (stmt: BasicBlockStmt δ) : Option (List (BasicBlockStmt δ)) :=
  match stmt with
  | .StmtOp op => do
    let op' ← replaceOpInOp op
    [BasicBlockStmt.StmtOp op']
  | .StmtAssign var idx op =>
      if var == nameMatch then
        some new_ops
      else do
        let op' ← replaceOpInOp op
        some [BasicBlockStmt.StmtAssign var idx op']
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
    match replaceOpInBBStmts concreteVarMatchName newOps prog with
    | some res => res
    | none => prog

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

def varDefInProg (t: T) :  List SSAVal := []
def varUseInProg (t: T) :  List SSAVal := []

theorem cons_is_append (head: T) (tail: List T) : head :: tail = [head] ++ tail
  := by rfl

def termResName (m: MTerm δ) : Option String :=
  match m with
  | .App .OP [ _, _, .App (.LIST .MOperand) 
        [.App .OPERAND [.Var _ ssaName .MSSAVal, _]] ] => some ssaName
  | _ => none

def getResName (stmt: BasicBlockStmt δ) : Option SSAVal :=
  match stmt with
  | .StmtAssign res _ _ => some res
  | .StmtOp _ => none

def run_split_head_tail [S: Semantics δ] :
  ∀ (pHead pTail) (env: SSAEnv δ), (run ⟦BasicBlock.mk "" [] (pHead ++ pTail)⟧ env) = 
    (run ⟦BasicBlock.mk "" [] pTail⟧ (run ⟦BasicBlock.mk "" [] pHead⟧ env).snd) := sorry

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

section MainTheorem
variable (δ: Dialect δα δσ δε)
         [S: Semantics δ]
         (σ: VarCtx δ)
         (mHead: List (MTerm δ))
         (mOrigin: MTerm δ)
         (mRes: List (MTerm δ))
         (headPat originPat resPat: List (BasicBlockStmt δ))
         (HHeadPat: MTerm.concretizeProg mHead σ = some headPat)
         (HOriginPat: MTerm.concretizeProg [mOrigin] σ = some originPat)
         (HResPat: MTerm.concretizeProg mRes σ = some resPat)
         (HRef: (∀ env, refinement (run ⟦BasicBlock.mk "" [] (headPat ++ originPat)⟧ env)
                                   (run ⟦BasicBlock.mk "" [] (headPat ++ resPat)⟧ env)))
         (origName: String)
         (HOrigName: termResName mOrigin = some origName)
         (prog: List (BasicBlockStmt δ))
         (Hmatch: (headPat ++ originPat).all (fun op => isOpInBBStmts op prog))

def main_theorem :
  ∀ (p: List (BasicBlockStmt δ)),
  ∀ ctx, (singleBBRegionStmtsObeySSA p ctx).isSome →
  ∀ (env: SSAEnv δ),
  (∀ val, ctx.isValDefined val →
      ∀ op, getDefiningOpInBBStmts val prog = some op →
      postSSAEnv op env) ->
  ∀ resP, replaceOpInBBStmts headName resPat p = some resP →
  refinement (run ⟦BasicBlock.mk "" [] p⟧ env) 
             (run ⟦BasicBlock.mk "" [] resP⟧ env)
  := by
    intros p
    induction p
    -- We do an induction over the program we are rewriting
    case nil =>
      -- The base case is easy, we couldn't find the operation in the program,
      -- thus we have a contradiction
      intros _ _ _ _ resP HresP
      simp [replaceOpInBBStmts] at HresP
    
    -- Induction case
    case cons head tail Hind =>
      intros ctx HSSA env HCtx resP HresP
      -- We first do a case analysis if we have rewritten or not the head op of the program
      simp [replaceOpInBBStmts] at HresP
      cases Hreplace: replaceOpInBBStmt headName resPat head
        <;> rw [Hreplace] at HresP <;> simp at HresP
      
      -- Here, the first operation has not been rewritten
      case none =>
        -- We first get the information that the rewrite must have worked in
        -- the tail of the program
        cases HreplaceTail: replaceOpInBBStmts headName resPat tail
          <;> rw [HreplaceTail] at HresP <;> simp at HresP
          <;> try contradiction
        rename_i resTail
        simp [bind, Option.bind] at HresP
        subst resP

        -- Then, we rewrite the `run` showing that we are first running the first statement,
        -- and then the tail
        rw [cons_is_append head tail]
        rw [cons_is_append head resTail]
        rw [run_split_head_tail]
        rw [run_split_head_tail]

        -- We get the dominance context for the tail
        simp [singleBBRegionStmtsObeySSA] at HSSA
        cases HSSAHead: singleBBRegionStmtObeySSA head ctx
          <;> rw [HSSAHead] at HSSA <;> try contradiction
        rename_i tailCtx
        simp [Option.bind] at HSSA
        specialize (Hind tailCtx HSSA)

        -- Running the first statement becomes an environment env'
        -- This is the environment that we are going to use for our induction
        generalize Henv': (run ⟦ BasicBlock.mk "" [] [head] ⟧ env).snd = env'
        specialize Hind env'

        -- We prove that if we added an SSAValue in the context, then the defining
        -- operation had to execute it.
        specialize Hind (by
          intros val Hval op HopInProg
          sorry
        )
        
        specialize Hind _ HreplaceTail
        assumption

      -- In this case, the first operation, or one operation of its region, was rewritten.
      case some resHeadP => 
        subst resP
        rw [cons_is_append head tail]
        rw [run_split_head_tail]
        rw [run_split_head_tail]
        sorry

end MainTheorem