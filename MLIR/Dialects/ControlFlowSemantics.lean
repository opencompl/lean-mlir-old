import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
open MLIR.AST

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
  | Op.mk "dummy.dummy" _ _ _ _ (.fn (.tuple []) (.int sgn sz)) => some do
      let i ← Fitree.trigger DummyE.Dummy
      SSAEnv.set? (δ := Gδ) (.int sgn sz) ret (.ofInt sgn sz i)
      return BlockResult.Next
  | Op.mk "dummy.true" _ _ _ _ (.fn (.tuple []) (.int sgn sz)) => some do
      let i ← Fitree.trigger DummyE.True
      SSAEnv.set? (δ := Gδ) (.int sgn sz) ret (.ofInt sgn sz i)
      return BlockResult.Next
  | Op.mk "dummy.false" _ _ _ _ (.fn (.tuple []) (.int sgn sz)) => some do
      let i ← Fitree.trigger DummyE.False
      SSAEnv.set? (δ := Gδ) (.int sgn sz) ret (.ofInt sgn sz i)
      return BlockResult.Next
  | _ => none

def DummyE.handle {E}: DummyE ~> Fitree E :=
  fun _ e =>
    match e with
    | .Dummy => return 42
    | .True => return 1
    | .False => return 0

instance: Semantics dummy where
  E := DummyE
  semantics_op := dummy_semantics_op
  handle := DummyE.handle

/-
### Dialect: `cf`
-/

instance cf: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

-- Since basic block branching semantics are currently implemented at the
-- language level, we have little to do in this dialect; branching instructions
-- just use the existing definitions.
inductive ControlFlowOp: Type → Type :=
  | Assert: (cond: FinInt 1) → (msg: String) → ControlFlowOp Unit

def cf_semantics_op (ret_name: Option SSAVal):
      Op Gδ → Option (Fitree (SSAEnvE Gδ +' ControlFlowOp) (BlockResult Gδ))
  | Op.mk "cf.br" [] [bbname] [] _ _ => some do
      return BlockResult.Branch bbname []
  | Op.mk "cf.condbr" [vcond] [bbtrue, bbfalse] _ _ _ => some do
      let condval <- Fitree.trigger $ SSAEnvE.Get (δ := Gδ) .i1 vcond
      return BlockResult.Branch
        (if condval.toUint != 0 then bbtrue else bbfalse) []
  | Op.mk "cf.ret" args [] [] _ (.fn (.tuple τs) _) =>
      if args.length = τs.length then
        some $ return BlockResult.Ret (List.zip args τs)
      else none
  | Op.mk "cf.assert" [arg] [] [] attrs (.fn (.tuple [.i1]) .unit) =>
      match attrs.find "msg" with
      | some (.str str) => some do
        let arg <- Fitree.trigger $ SSAEnvE.Get (δ := Gδ) .i1 arg
        Fitree.trigger $ ControlFlowOp.Assert arg str
        return BlockResult.Next
      | none => some do
        let arg <- Fitree.trigger $ SSAEnvE.Get (δ := Gδ) .i1 arg
        Fitree.trigger $ ControlFlowOp.Assert arg "<assert failed>"
        return BlockResult.Next
      | _ => none
  | _ => none

-- Default pure handler
def ControlFlowOp.handle {E}: ControlFlowOp ~> Fitree E
  | _, Assert cond msg =>
    return ()

-- Alternative handler in WriterT (with output)
def ControlFlowOp.handleLogged {E}: ControlFlowOp ~> WriterT (Fitree E)
  | _, Assert cond msg => do
    if cond.toUint == 0 then
      logWriterT msg
    return ()

instance: Semantics cf where
  E := ControlFlowOp
  semantics_op := cf_semantics_op
  handle := ControlFlowOp.handle

/-
### Examples and testing
-/

def run_dummy_cf_region: Region (dummy + cf) → String := fun r =>
  runLogged (semantics_region 99 r) SSAEnv.empty |>.fst |>.snd

def run_dummy_cf_region': Region (dummy + cf) → String := fun r =>
  let t := semantics_region 99 r
  let t := interp_ub! t
  let t := interp_ssa t SSAEnv.empty
  let t: Fitree ControlFlowOp _ := interp (Fitree.case_ DummyE.handle
    (fun _ e => Fitree.trigger e: ControlFlowOp ~> Fitree _)) t
  let t: WriterT (Fitree PVoid) _ := interp ControlFlowOp.handleLogged t
  t.run.run.snd

--

def dummy_stmt: BasicBlockStmt dummy := [mlir_bb_stmt|
  %dummy = "dummy.dummy" () : () -> i32
]

def true_stmt: BasicBlockStmt dummy := [mlir_bb_stmt|
  %true = "dummy.true" () : () -> i1
]

def false_stmt: BasicBlockStmt dummy := [mlir_bb_stmt|
  %false = "dummy.false" () : () -> i1
]

def ex_branch_true: Region dummy := [mlir_region| {
  ^entry:
    %x = "dummy.true" () : () -> i1
    "cf.condbr"(%x) [^bbtrue, ^bbfalse] : ()

  ^bbtrue:
    %y = "dummy.dummy" () : () -> i32
    "cf.ret" () : () -> ()

  ^bbfalse:
    %z = "dummy.dummy" () : () -> i32
    "cf.ret" () : () -> ()
}]

#eval ex_branch_true
#eval run_dummy_cf_region ex_branch_true

def ex_branch_false := [mlir_region| {
  ^entry:
    %x = "dummy.false" () : () -> i1
    "cf.condbr"(%x) [^bbtrue, ^bbfalse] : ()

  ^bbtrue:
    %y = "dummy.dummy" () : () -> i32
    "cf.ret" () : () -> ()

  ^bbfalse:
    %z = "dummy.dummy" () : () -> i32
    "cf.ret" () : () -> ()
}]

#eval ex_branch_false
#eval run_dummy_cf_region ex_branch_false

def ex_assert_true := [mlir_region| {
  %true = "dummy.true" (): () -> i1
  "cf.assert" (%true) {msg = "is false!"}: (i1) -> ()
  "cf.ret" (): () -> ()
}]

def ex_assert_false := [mlir_region| {
  %false = "dummy.false" (): () -> i1
  "cf.assert" (%false) {msg = "is false!"}: (i1) -> ()
  "cf.ret" (): () -> ()
}]

-- assert is fine: prints nothing
#eval run_dummy_cf_region' ex_assert_true
-- assert fails: prints an error
#eval run_dummy_cf_region' ex_assert_false
