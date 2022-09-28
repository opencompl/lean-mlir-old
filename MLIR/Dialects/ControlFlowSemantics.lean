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

def dummy_semantics_op: IOp Δ →
      Fitree (RegionE Δ +' UBE) (BlockResult Δ)
  | IOp.mk "dummy.dummy" [.int sgn sz] [] _ _ _ => do
      let i := 42
      return BlockResult.Next ⟨.int sgn sz, FinInt.ofInt sz i⟩
  | IOp.mk "dummy.true" [.int sgn sz] [] _ _ _ => do
      let i := 1
      return BlockResult.Next ⟨.int sgn sz, FinInt.ofInt sz i⟩
  | IOp.mk "dummy.false" [.int sgn sz] [] _ _ _ => do
      let i := 0
      return BlockResult.Next ⟨.int sgn sz, FinInt.ofInt sz i⟩
  | _ => Fitree.trigger (UBE.Unhandled)


instance: Semantics dummy where
  semantics_op := dummy_semantics_op

/-
### Dialect: `cf`
-/

instance cf: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

-- TODO: allow this to be given by `IOp cf`
def cfSemanticsOp: IOp Δ →
      (Fitree (RegionE Δ +' UBE) (BlockResult Δ))
  | IOp.mk "cf.br" _ [] [bbname] 0 _ => do
      return BlockResult.Branch bbname []
  | IOp.mk "cf.condbr" _ [⟨.i1, condval⟩] [bbtrue, bbfalse] _ _ => do
      return BlockResult.Branch
        (if condval.toUint != 0 then bbtrue else bbfalse) []
  | IOp.mk "cf.ret" _ args [] 0 _ =>
       return BlockResult.Ret args
  | IOp.mk "cf.assert" _ [⟨.i1, arg⟩] [] 0 attrs =>
      match attrs.find "msg" with -- TODO: convert this to a pattern match.
      | some (.str str) => do
             -- Fitree.trigger $ UBE.UB (.some s!"{arg} {str}")
             return BlockResult.Next ⟨.unit, ()⟩
      | _ => do
            Fitree.trigger $ UBE.UB (.some s!"{arg} <assert failed>")
            return BlockResult.Next ⟨.unit, ()⟩
  | _ => Fitree.trigger $ UBE.Unhandled


instance: Semantics cf where
  semantics_op := cfSemanticsOp

/-
### Examples and testing
-/

def run_dummy_cf_region: Region (dummy + cf) → String := fun r =>
  match runLogged (semanticsRegion 99 r []) SSAEnv.empty with
  | .error msg => msg
  | .ok ((_, log), _) => log

def run_dummy_cf_region': Region (dummy + cf) → String := fun r =>
  let t := semanticsRegion 99 r []
  let t := interpSSALogged' t SSAEnv.empty
  -- let t: Fitree (ControlFlowE +' UBE) _ :=
  --   t.interp (Fitree.case
  --     (Fitree.case DummyE.handle (fun _ e => Fitree.trigger e))
  --     (fun _ e => Fitree.trigger e))
  -- let t: WriterT (Fitree UBE) _ :=
  --   t.interp (Fitree.case ControlFlowE.handleLogged Fitree.liftHandler)
  let t := interpUB t
  match Fitree.run t with
  | .error msg => msg
  | .ok (((_, log), _)) => log

--

def dummy_stmt: Op dummy := [mlir_op|
  %dummy = "dummy.dummy" () : () -> i32
]

def true_stmt: Op dummy := [mlir_op|
  %true = "dummy.true" () : () -> i1
]

def false_stmt: Op dummy := [mlir_op|
  %false = "dummy.false" () : () -> i1
]

def ex_branch_true: Region dummy := [mlir_region| {
  ^entry:
    %x = "dummy.true" () : () -> i1
    "cf.condbr"(%x) [^bbtrue, ^bbfalse] : (i1) -> ()

  ^bbtrue:
    %y = "dummy.dummy" () : () -> i32
    "cf.ret" () : () -> ()

  ^bbfalse:
    %z = "dummy.dummy" () : () -> i32
    "cf.ret" () : () -> ()
}]

#eval ex_branch_true
#eval run_dummy_cf_region ex_branch_true

def ex_branch_false : Region dummy := [mlir_region| {
  ^entry:
    %x = "dummy.false" () : () -> i1
    "cf.condbr"(%x) [^bbtrue, ^bbfalse] : (i1) -> ()

  ^bbtrue:
    %y = "dummy.dummy" () : () -> i32
    "cf.ret" () : () -> ()

  ^bbfalse:
    %z = "dummy.dummy" () : () -> i32
    "cf.ret" () : () -> ()
}]

#eval ex_branch_false
#eval run_dummy_cf_region ex_branch_false

def ex_assert_true : Region dummy := [mlir_region| {
  %true = "dummy.true" (): () -> i1
  "cf.assert" (%true) {msg = "is false!"}: (i1) -> ()
  "cf.ret" (): () -> ()
}]

def ex_assert_false : Region dummy := [mlir_region| {
  %false = "dummy.false" (): () -> i1
  "cf.assert" (%false) {msg = "is false!"}: (i1) -> ()
  "cf.ret" (): () -> ()
}]

-- assert is fine: prints nothing
#eval run_dummy_cf_region' ex_assert_true
-- assert fails: prints an error
#eval run_dummy_cf_region' ex_assert_false

/-
### Theorems
-/

namespace cf_th1
def LHS: Region cf := [mlir_region|
{
  ^entry:
    %x = "dummy.true" () : () -> i1
    "cf.condbr"(%x) [^bbtrue, ^bbfalse] : (i1) -> ()

  ^bbtrue:
    %y = "dummy.dummy" () : () -> i32
    "cf.ret" () : () -> ()

  ^bbfalse:
    %z = "dummy.dummy" () : () -> i32
    "cf.ret" () : () -> ()
}]

-- | this is mildly janky, but meh.
def RHS (bb_true: BasicBlock cf): Region cf :=
  Region.mk (δ := cf) bb_true

-- TODO: Proof of equivalence theorem for true-if in the cf dialect
end cf_th1
