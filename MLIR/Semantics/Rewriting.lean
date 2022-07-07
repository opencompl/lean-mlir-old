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

def splitHeadTail (l: List T) : Option (List T × T) :=
  match l with
  | [] => none
  | [t] => some ([], t)
  | t::l' => match splitHeadTail l' with
             | some (head, tail) => some (t::head, tail)
             | none => none

def postSSAEnv [Semantics δ] (op: BasicBlockStmt δ) (env: SSAEnv δ) : Prop :=
  ∃ env', (run ⟦op⟧ env').snd = env

def postSSAEnvList [Semantics δ] (op: List (BasicBlockStmt δ)) (env: SSAEnv δ) : Prop :=
  ∃ env', (run ⟦BasicBlock.mk "" [] op⟧ env').snd = env

def getResName (stmt: BasicBlockStmt δ) : Option SSAVal :=
  match stmt with
  | .StmtAssign res _ _ => some res
  | .StmtOp _ => none

def getOp (stmt: BasicBlockStmt δ) : Op δ :=
  match stmt with
  | .StmtAssign _ _ op => op
  | .StmtOp op => op

def stmtHasNoRegions (stmt: BasicBlockStmt δ) : Bool :=
  match stmt with
  | .StmtOp (.mk _ _ _ [] _ _) => true
  | .StmtAssign _ _ (.mk _ _ _ [] _ _) => true
  | _ => false

def stmtsHaveNoRegions (stmts: List (BasicBlockStmt δ)) : Bool :=
  match stmts with
  | [] => true
  | stmt::stmts' => stmtHasNoRegions stmt && stmtsHaveNoRegions stmts'

instance [Monad m] [LawfulMonad m] : LawfulMonad (OptionT m) where
  id_map         := by sorry
  map_const      := by sorry
  seqLeft_eq     := by sorry
  seqRight_eq    := by sorry
  pure_seq       := by sorry
  bind_pure_comp := by sorry
  bind_map       := by sorry
  pure_bind      := by sorry
  bind_assoc     := by sorry;

instance : LawfulMonad (Fitree F) where
  id_map         := by sorry
  map_const      := by sorry
  seqLeft_eq     := by sorry
  seqRight_eq    := by sorry
  pure_seq       := by sorry
  bind_pure_comp := by sorry
  bind_map       := by sorry
  pure_bind      := by sorry
  bind_assoc     := by sorry

theorem interp_bind [Monad M] [LawfulMonad M] (h: E ~> M) (t: Fitree E A) (k: A -> Fitree E B):
  interp h (Fitree.bind t k) = bind (interp h t) (fun x => interp h (k x)) := by {
  induction t;
  case Ret monadInstanceM r => {
    simp [interp, bind, Fitree.bind];
  }
  case Vis lawful T' e' k' IND => {
      simp[interp, bind, Fitree.bind, IND];
  }
}

def opArgsFitree {Δ: Dialect α σ ε} [S: Semantics Δ] (args: List SSAVal) (τs: List (MLIRType Δ)) 
 : Fitree (UBE +' SSAEnvE Δ +' S.E) (List ((τ : MLIRType Δ) × MLIRType.eval τ))
   := ((List.zip args τs).mapM (fun (name, τ) => do
    return ⟨τ, ← Fitree.trigger <| SSAEnvE.Get τ name⟩))

theorem run_preserves_env_set {δ: Dialect α σ ε} [Semantics δ] (stmt: BasicBlockStmt δ) (env env': SSAEnv δ) :
    ∀ r, run ⟦ stmt ⟧ env = (some r, env') →
    stmtHasNoRegions stmt →
    ∀ name, getResName stmt ≠ some name →
    name ∉ (getOp stmt).args → 
    ∀ τ v, run ⟦ stmt ⟧ (SSAEnv.set name τ v env) = (some r, SSAEnv.set name τ v env') := by
  intros r HRun HNoRegions name HNameRes HNameArgs τ v
  simp [run, Denote.denote]
  simp [run, Denote.denote] at HRun
  unfold denoteBBStmt
  unfold denoteBBStmt at HRun
  cases stmt
  case StmtAssign res ix op => sorry
  case StmtOp op =>
    unfold denoteOp
    unfold denoteOp at HRun
    split
    -- Non-UB case
    . rename_i name args bbs regions args res
      simp
      simp at HRun
      simp_itree
      simp [interp_ub, interp_ssa, interp_state]
      rw [interp_bind]
      sorry
    -- UB case because type does not match. We have a contradiction
    . rename_i Hop
      revert HRun
      split
      . rename_i oName oArgs oBbs oRegions oAttrs oArgsTy oResTy
        specialize (Hop oName oArgs oBbs oRegions oAttrs oArgsTy oResTy)
        contradiction
      . simp [Fitree.run, interp_ssa, interp_ub, interp_state, interp, pure, Fitree.ret]
      

def postSSAEnv_preserves [Semantics δ] (stmt: BasicBlockStmt δ) (env: SSAEnv δ) :
    postSSAEnv stmt env → 
    ∀ name, getResName stmt ≠ some name →
    name ∉ (getOp stmt).args → 
    postSSAEnv stmt (env.set name τ v) := by
  intros HPost name HNameNeStmtName HNameNotInArgs 
  unfold postSSAEnv at *
  have ⟨env', HRunStmt⟩ := HPost
  exists (env'.set name τ v)
  apply run_preserves_env_set <;> assumption
  

def varDefInProg (t: T) :  List SSAVal := []
def varUseInProg (t: T) :  List SSAVal := []

theorem cons_is_append (head: T) (tail: List T) : head :: tail = [head] ++ tail
  := by rfl

def termResName (m: MTerm δ) : Option String :=
  match m with
  | .App .OP [ _, _, .App (.LIST .MOperand) 
        [.App .OPERAND [.Var _ ssaName .MSSAVal, _]] ] => some ssaName
  | _ => none

theorem preserves_refinement [Semantics δ] (bb: BasicBlock δ): 
    SSAEnv.refinement env env' →
    refinement (run ⟦bb⟧ env) (run ⟦bb⟧ env') := by
  sorry

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

theorem no_regions_implies_no_replace (stmt: BasicBlockStmt δ) :
    stmtHasNoRegions stmt →
    ∀ val new_ops res, replaceOpInBBStmt val new_ops stmt = some res →
    (res = new_ops) ∧ (∃ ix op, stmt = BasicBlockStmt.StmtAssign val ix op) := by
  sorry

def getOneUser (val: SSAVal) (prog: List (BasicBlockStmt δ)) : Option (BasicBlockStmt δ) :=
  match prog with
  | [] => none
  | stmt::prog' =>
    if val ∈ (getOp stmt).args then
      some stmt
    else
      getOneUser val prog'

def eqStmt (stmt stmt': BasicBlockStmt δ) : Bool :=
  match stmt, stmt' with
  | .StmtAssign res _ _, .StmtAssign res' _ _ => res == res'
  | _, _ => false

def isRootedBy (stmt: BasicBlockStmt δ) (prog: List (BasicBlockStmt δ))
               (root: BasicBlockStmt δ) 
               (fuel: Nat) : Bool :=
  match fuel with
  | 0 => false
  | .succ fuel' =>
    match stmt with
    | .StmtAssign res _ _ =>
      match getOneUser res prog with
      | none => false
      | some parentStmt =>
          eqStmt parentStmt root || isRootedBy parentStmt prog root fuel'
    | _ => false
      

def allRootedToLast (prog: List (BasicBlockStmt δ)) : Bool :=
  let headTail := splitHeadTail prog
  match headTail with
  | none => false
  | some (stmts', root) =>
    stmts'.all (fun stmt => isRootedBy stmt prog root (prog.length))

abbrev recursiveContext (ctx: DomContext δ) (p: List (BasicBlockStmt δ)) : Prop :=
  (∀ val, ctx.isValDefined val →
    ∀ op, getDefiningOpInBBStmts val p = some op →
    ∀ operand, operand ∈ (getOp op).args →
    ctx.isValDefined operand)

theorem isRootedBy_implies_isValDefined :
    ∀ (p: List (BasicBlockStmt δ)) (ctx: DomContext δ),
    recursiveContext ctx p →
    ∀ stmt, stmt ∈ p →
    ∀ resName, getResName stmt = some resName →
    ∀ root, root ∈ p →
    (∀ val, val ∈ (getOp root).args → ctx.isValDefined val) →
    ∀ fuel, isRootedBy stmt p root fuel →
    ctx.isValDefined resName := by
  intros p ctx HRecCtx stmt HStmtInP resName HResName root HRoot HRootOperands fuel
  revert root resName stmt
  induction fuel <;> try (intros; contradiction)
  intros stmt HStmtInP resName HResName root HRoot HRootOperands HRooted
  sorry

theorem StmtOp_obeys_ssa_some_no_change (headOp: Op δ) (ctx: DomContext δ):
    (singleBBRegionStmtObeySSA (BasicBlockStmt.StmtOp headOp) ctx).isSome →
    singleBBRegionStmtObeySSA (BasicBlockStmt.StmtOp headOp) ctx = some ctx := by
  sorry

def opGetResType (op: Op δ) : Option (MLIRType δ):=
  match op with
  | .mk _ _ _ _ _ (MLIRType.fn _ (MLIRType.tuple [τ])) => some τ
  | _ => none

theorem stmt_obeys_ssa_implies_res_type (stmt: BasicBlockStmt δ) (ctx: DomContext δ):
  (singleBBRegionStmtObeySSA stmt ctx).isSome →
  (opGetResType (getOp stmt)).isSome := by sorry

theorem StmtAssign_obeys_ssa_some (res: SSAVal) ix (op: Op δ) (ctx: DomContext δ):
    (singleBBRegionStmtObeySSA (BasicBlockStmt.StmtAssign res ix op) ctx).isSome →
    ∃ τ, opGetResType op = some τ ∧
         singleBBRegionStmtObeySSA (BasicBlockStmt.StmtAssign res ix op) ctx = some (ctx.addVal res τ):= by sorry

section MainTheorem
variable (δ: Dialect δα δσ δε)
         [S: Semantics δ]
         (headPat : List (BasicBlockStmt δ))
         (originPat: BasicBlockStmt δ)
         (resPat: List (BasicBlockStmt δ))
         (HRoot: allRootedToLast (headPat ++ [originPat]))
         (HRef: (∀ env, refinement (run ⟦BasicBlock.mk "" [] (headPat ++ [originPat])⟧ env)
                                   (run ⟦BasicBlock.mk "" [] (headPat ++ resPat)⟧ env)))
         (origName: String)
         (HOrigName: getResName originPat = some origName)
         (prog: List (BasicBlockStmt δ))
         (Hmatch: (headPat ++ [originPat]).all (fun op => isOpInBBStmts op prog))


def main_theorem_stmt :
  ∀ (p: List (BasicBlockStmt δ)),
  stmtsHaveNoRegions p →
  ∀ ctx, (singleBBRegionStmtsObeySSA p ctx).isSome →
  ∀ (env: SSAEnv δ),
  recursiveContext ctx prog →
  (∀ val, ctx.isValDefined val →
      ∀ op, getDefiningOpInBBStmts val prog = some op →
        postSSAEnv op env) →
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
      intros _ _ _ _ _ _ resP HresP
      simp [replaceOpInBBStmts] at HresP
    
    -- Induction case
    case cons head tail Hind =>
      intros HNoRegions ctx HSSA env HCtx HCtxRec resP HresP
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

        -- The tail still has no regions
        simp [stmtsHaveNoRegions] at HNoRegions
        have ⟨HNoRegionsHead, HNoRegionsTail⟩ := HNoRegions
        specialize Hind HNoRegionsTail

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

        -- We prove that if we added an SSAValue in the context, then the operands of
        -- its definition were already in the context as well.
        have HRec : recursiveContext tailCtx prog := by
          cases HHead: head <;> subst head

          -- StmtOp case, we do not change the context, and thus this is trivial
          case StmtOp headOp =>
            rw [StmtOp_obeys_ssa_some_no_change] at HSSAHead <;> 
              try (rw [HSSAHead]; rfl)
            simp at HSSAHead
            subst tailCtx
            assumption

          -- StmtOp case, we do change the context, and since we know that the program
          -- obeys SSA, then operands have to be in the context.
          case StmtAssign resHead ixHead opHead =>
            have HTailCtx := (StmtAssign_obeys_ssa_some _ _ _ _ (by rw [HSSAHead]; rfl))
            have ⟨τ, ⟨HOpHeadType, HTailCtx⟩⟩ := HTailCtx
            rw [HTailCtx] at HSSAHead; simp at HSSAHead; subst tailCtx
            sorry

        specialize Hind HRec

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

        -- Since we interpret both heads and then the common tail, we just need to
        -- prove that the interpretation of both heads will result in a refinement,
        -- since then the interpretation of the same tail program will preserve that
        -- refinement
        rw [cons_is_append head tail]
        rw [run_split_head_tail]
        rw [run_split_head_tail]
        apply preserves_refinement

        -- The operation has to be an StmtAssign, because we have done a rewrite,
        -- and the rewrite was done on an stmt with no regions
        simp [stmtsHaveNoRegions] at HNoRegions
        have ⟨HNoRegionsHead, HNoRegionsTail⟩ := HNoRegions
        have HHead := no_regions_implies_no_replace _ HNoRegionsHead _ _ _ Hreplace
        have ⟨_, ⟨ix, headOp, _⟩⟩ := HHead
        subst resHeadP
        subst head
        
        sorry
end MainTheorem