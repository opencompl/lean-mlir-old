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

instance scf: Dialect Void Void (fun x => Unit) where
  iα := inferInstance
  iε := inferInstance

-- | interesting, what is the type of ScfFor?
-- | TODO: use the return type. For now, just do unit.
inductive ScfE: Type → Type :=
  | For: (low:Int) → (upper: Int) → (step: Int)  → ScfE Unit




-- | run a loop, decrementing i from n to -
-- | ix := lo + (n - i) * step
def run_loop_bounded_go [Monad m] (n: Nat) (i: Nat) (lo: Int) (step: Int)
  (accum: a) (eff: Int -> a -> m a): m a := do
   let ix : Int := lo + (n - i) * step
   let accum <- eff ix accum
   match i with
   | .zero => return accum
   | .succ i' => run_loop_bounded_go n i' lo step accum eff

-- | TODO: use this to run regions.
def run_loop_bounded [Monad m] (n: Nat) (lo: Int) (step: Int) (accum: a) (eff: Int -> a -> m a): m a := 
  run_loop_bounded_go n n lo step accum eff

#check semantics_region_single_bb

#check ScfE.For
-- | TODO: refactor to (1) an effect, (2) an interpretation
def scf_semantics_op {Gα Gσ Gε} {Gδ: Dialect Gα Gσ Gε} [S: Semantics Gδ]
  (ret_name: Option SSAVal):
        Op Gδ → Option (Fitree (UBE +' SSAEnvE Gδ +' S.E Gδ +' ScfE) (BlockResult Gδ))
  | Op.mk "scf.for" [lo, hi, step] _ [r] _ (.fn (.tuple []) (.int sgn sz)) => some do
      let lo : FinInt sz <- SSAEnv.get? Gδ (MLIRType.int sgn sz) lo
      let hi : FinInt sz <- SSAEnv.get? Gδ (MLIRType.int sgn sz) hi
      let step : FinInt sz <- SSAEnv.get? Gδ (MLIRType.int sgn sz) step
      -- let rsem := semantics_region_single_bb r
      let t <- Fitree.trigger (ScfE.For (FinInt.toSint' lo) (FinInt.toSint' hi) (FinInt.toSint' step));
      SSAEnv.set? (δ := Gδ) MLIRType.unit ret_name ()
      -- let nsteps : Int := ((FinInt.toSint'  hi) - (FinInt.toSint' lo)) / FinInt.toSint' step
      -- let out <- run_loop_bounded (a := PUnit) 
      --            (n := nsteps.toNat)
      --            (lo := (FinInt.toSint' lo))
      --            (step := (FinInt.toSint' step)) 
      --            (accum := PUnit.unit)
      --            (eff := (fun i _ => (semantics_region_single_bb r))) -- how to type this correctly?
                 -- (eff := (fun i _ => pure PUnit.unit))
      -- let i ← Fitree.trigger (ScfE.For 0 0 0)
      -- SSAEnv.set? (δ := Gδ) (.int sgn sz) ret (.ofInt sgn sz i)
      return BlockResult.Next
  | _ => none

def handle_scf {E}: ScfE ~> Fitree E :=
  fun _ e =>
    match e with
    | .For lo hi step => do
      let nsteps : Int := (hi - lo) / step
      let out <- run_loop_bounded (a := PUnit) 
                 (n := nsteps.toNat)
                 (lo := lo)
                 (step := step)
                 (accum := PUnit.unit)
                 (eff := (fun i _ => (semantics_region_single_bb r))) -- how to type this correctly?
                 -- (eff := (fun i _ => pure PUnit.unit))


instance: Semantics sef where
  E := ScfE
  semantics_op := scf_semantics_op
  handle := handle_scf

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
