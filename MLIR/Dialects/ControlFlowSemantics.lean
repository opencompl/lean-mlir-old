import MLIR.Semantics.Fitree
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.InvalidOp
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
open MLIR.AST

/-
### General semantics
-/

inductive BlockResult {Gα Gσ Gε} (Gδ: Dialect Gα Gσ Gε)
| Branch (bb: BBName) (args: List SSAVal)
| Ret (rets: List (SSAVal × MLIRType Gδ))
| Next

class Semantics {α σ ε} (δ: Dialect α σ ε) where
  -- Events modeling the dialect's operations
  E: Type → Type

  -- Operation semantics function: maps an `Op` to an interaction tree. Usually
  -- this simply emits an event of `E` and records the return value into the
  -- environment, and could be automated.
  semantics_op {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε}:
    Option SSAVal → Op Gδ → Option (Fitree (SSAEnvE Gδ +' E) (BlockResult Gδ))

  -- TODO: Allow a dialects' semantics to specify their terminators along with
  -- TODO| their branching behavior, instead of hardcoding it for cf

  -- Event handler used when interpreting the operations and running programs.
  -- This is where most of the semantics and computations take place.
  -- TODO: Allow dialect handlers to emit events into other dialects
  handle: E ~> Fitree PVoid

instance {α₁ σ₁ ε₁} {δ₁: Dialect α₁ σ₁ ε₁} {α₂ σ₂ ε₂} {δ₂: Dialect α₂ σ₂ ε₂}
    [S₁: Semantics δ₁] [S₂: Semantics δ₂]: Semantics (δ₁ + δ₂) where
  E := S₁.E +' S₂.E
  semantics_op ret_name op :=
    (S₁.semantics_op ret_name op).map (.translate Member.inject) <|>
    (S₂.semantics_op ret_name op).map (.translate Member.inject)
  handle := Fitree.case_ S₁.handle S₂.handle

def semantics_op! {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]:
    Option SSAVal → Op Gδ →
    Fitree (InvalidOpE Gδ +' SSAEnvE Gδ +' S.E) (BlockResult Gδ) :=
  fun ret op =>
    match S.semantics_op ret op with
    | some t => t.translate Member.inject
    | none => do Fitree.trigger (InvalidOpE.InvalidOp op); return .Next

def semantics_bbstmt {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]:
    BasicBlockStmt Gδ →
    Fitree (InvalidOpE Gδ +' SSAEnvE Gδ +' S.E) (BlockResult Gδ)
| .StmtAssign val _ op => semantics_op! (some val) op
| .StmtOp op => semantics_op! none op

/-
### Dialect: `dummy`
-/

instance dummy: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

inductive DummyE: Type → Type :=
  | Dummy: DummyE Int
  | True: DummyE Int
  | False: DummyE Int

def dummy_semantics_op {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} (ret: Option SSAVal):
      Op Gδ → Option (Fitree (SSAEnvE Gδ +' DummyE) (BlockResult Gδ))
  | Op.mk "dummy.dummy" _ _ _ _ (.fn (.tuple []) (.int b)) => some do
      let i ← Fitree.trigger DummyE.Dummy
      SSAEnv.set? (δ := Gδ)  (.int b) ret i
      return BlockResult.Next
  | Op.mk "dummy.true" _ _ _ _ (.fn (.tuple []) (.int b)) => some do
      let i ← Fitree.trigger DummyE.True
      SSAEnv.set? (δ := Gδ) (.int b) ret i
      return BlockResult.Next
  | Op.mk "dummy.false" _ _ _ _ (.fn (.tuple []) (.int b)) => some do
      let i ← Fitree.trigger DummyE.False
      SSAEnv.set? (δ := Gδ) (.int b) ret i
      return BlockResult.Next
  | _ => none

def handle_dummy {E}: DummyE ~> Fitree E :=
  fun _ e =>
    match e with
    | .Dummy => return 42
    | .True => return 1
    | .False => return 0

instance: Semantics dummy where
  E := DummyE
  semantics_op := dummy_semantics_op
  handle := handle_dummy

/-
### Dialect: `cf`
-/

instance cf: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

set_option hygiene false in
genInductive ControlFlowOp #[
  OpSpec.mk "cf.br" `(
    (target: String) →
    (args: List ((τ: MLIRTy) × τ.eval)) →
    ControlFlowOp Unit)
  , OpSpec.mk "cf.condbr" `(
    (cond: Bool)  →
    ControlFlowOp Unit)
  -- , OpSpec.mk "runRegion" `(
  --    (region: Region) →
  --    (τ: MLIRTy) →
  --    ControlFlowOp (τ.eval))
  ]

-- | interpret branch and conditional branch
def cf_semantics_op (ret_name: Option SSAVal):
      Op Gδ → Option (Fitree (SSAEnvE Gδ +' PVoid) (BlockResult Gδ))
  | Op.mk "cf.br" [] [bbname] [] _ _ => some do
      return BlockResult.Branch bbname []
  | Op.mk "cf.condbr" [vcond] [bbtrue, bbfalse] _ _ _ => some do
      let condval <- Fitree.trigger $ SSAEnvE.Get (δ := Gδ) (.int 1) vcond
      return BlockResult.Branch (if condval != 0 then bbtrue else bbfalse) []
  | Op.mk "cf.ret" args [] [] _ (.fn (.tuple τs) _) =>
      if args.length = τs.length then
        some $ return BlockResult.Ret (List.zip args τs)
      else none
  | _ => none

instance: Semantics cf where
  E := PVoid
  semantics_op := cf_semantics_op
  handle := fun _ e => nomatch e

/-
### More general semantics
-/

-- TODO: Add the basic block arguments and bind them before running the block
def semantics_bb {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (bb: BasicBlock Gδ):
    Fitree (InvalidOpE Gδ +' SSAEnvE Gδ +' S.E) (BlockResult Gδ) := do
  -- TODO: we assume all statements return BlockResult.Next except the last
  for stmt in bb.stmts.init do
    let _ ← semantics_bbstmt stmt
  match bb.stmts.getLast? with
  | some stmt => semantics_bbstmt stmt
  | none => return BlockResult.Next

def semantics_region_go {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (fuel: Nat) (r: Region Gδ) (bb: BasicBlock Gδ):
    Fitree (InvalidOpE Gδ +' SSAEnvE Gδ +' S.E) (BlockResult Gδ) :=
  match fuel with
  | 0 => return .Next
  | fuel' + 1 => do
      match ← semantics_bb bb with
        | .Branch bbname args =>
            -- TODO: Pass the block arguments
            match r.getBasicBlock bbname with
            | some bb' => semantics_region_go fuel' r bb'
            | none => return .Next
        | .Ret rets => return .Ret rets
        | .Next => return .Next

-- TODO: Pass region arguments
-- TODO: Forward region's return type and value
def semantics_region {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    (fuel: Nat) (r: Region Gδ):
    Fitree (InvalidOpE Gδ +' SSAEnvE Gδ +' S.E) Unit := do
  let _ ← semantics_region_go fuel r (r.bbs.get! 0)

def semantics_run {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ] {R}
    (t: Fitree (InvalidOpE Gδ +' SSAEnvE Gδ +' S.E) R) (env: SSAEnv Gδ):
    R × SSAEnv Gδ :=
  let t := interp_invalid! t
  let t := interp_ssa t env
  let t := interp S.handle t
  t.run

def semantics_run_logged {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
    {R} (t: Fitree (InvalidOpE Gδ +' SSAEnvE Gδ +' S.E) R) (env: SSAEnv Gδ):
    (R × String) × SSAEnv Gδ :=
  let t := interp_invalid! t
  let t := (interp_ssa_logged t).run env
  let t := interp S.handle t
  t.run

/-
### Examples and testing
-/

-- #reduce spins on the dialect coercion because it's mutually inductive, even
-- with skipProofs := true (why?!), so define it directly in the dummy dialect
def dummy_stmt: BasicBlockStmt dummy := [mlir_bb_stmt|
  %dummy = "dummy.dummy"() : () -> i32
]
#reduce semantics_bbstmt dummy_stmt

def true_stmt: BasicBlockStmt dummy := [mlir_bb_stmt|
  %true = "dummy.true"() : () -> i1
]
#reduce semantics_bbstmt true_stmt

def false_stmt: BasicBlockStmt dummy := [mlir_bb_stmt|
  %false = "dummy.false"() : () -> i1
]
#reduce semantics_bbstmt false_stmt

def run_dummy_cf_region: Region (dummy + cf) → String := fun r =>
  semantics_run_logged (semantics_region 99 r) SSAEnv.empty |>.fst |>.snd

def ex_branch_true: Region dummy := [mlir_region| {
  ^entry:
    %x = "dummy.true"() : () -> i1
    "cf.condbr"(%x) [^bbtrue, ^bbfalse] : ()

  ^bbtrue:
    %y = "dummy.dummy" () : () -> i32
    "cf.ret" () : ()

  ^bbfalse:
    %z = "dummy.dummy" () : () -> i32
    "cf.ret" () : ()
}]

#eval ex_branch_true
#eval run_dummy_cf_region ex_branch_true

def ex_branch_false := [mlir_region| {
  ^entry:
    %x = "dummy.false"() : () -> i1
    "cf.condbr"(%x) [^bbtrue, ^bbfalse] : ()

  ^bbtrue:
    %y = "dummy.dummy" () : () -> i32
    "ret" () : ()

  ^bbfalse:
    %z = "dummy.dummy" () : () -> i32
    "ret" () : ()
}]

#eval ex_branch_false
#eval run_dummy_cf_region ex_branch_false
