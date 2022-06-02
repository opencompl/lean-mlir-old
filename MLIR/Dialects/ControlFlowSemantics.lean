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
      let condval <- Fitree.trigger $ SSAEnvE.Get (δ := Gδ) (.i1) vcond
      return BlockResult.Branch
        (if condval.toUint != 0 then bbtrue else bbfalse) []
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
### Examples and testing
-/

-- #reduce spins on the dialect coercion because it's mutually inductive, even
-- with skipProofs := true (why?!), so define it directly in the dummy dialect
def dummy_stmt: BasicBlockStmt dummy := [mlir_bb_stmt|
  %dummy = "dummy.dummy"() : () -> i32
]

def true_stmt: BasicBlockStmt dummy := [mlir_bb_stmt|
  %true = "dummy.true"() : () -> i1
]

def false_stmt: BasicBlockStmt dummy := [mlir_bb_stmt|
  %false = "dummy.false"() : () -> i1
]

def run_dummy_cf_region: Region (dummy + cf) → String := fun r =>
  runLogged (semantics_region 99 r) SSAEnv.empty |>.fst |>.snd

def ex_branch_true: Region dummy := [mlir_region| {
  ^entry:
    %x = "dummy.true"() : () -> i1
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
    %x = "dummy.false"() : () -> i1
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
