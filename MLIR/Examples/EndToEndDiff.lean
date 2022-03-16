import MLIR.AST
import MLIR.EDSL
import MLIR.PDL

open MLIR.AST
open MLIR.EDSL

inductive OpVerifier where
| MinArgs: Nat -> OpVerifier
| MaxArgs: Nat -> OpVerifier
| ExactArgs: Nat -> OpVerifier
-- | And: OpVerifier -> OpVerifier -> OpVerifier
-- | MinRegions: Nat -> OpVerifier
| T: OpVerifier



inductive InteractM (α: Type)
| ok: (val: α) ->  InteractM α
| error: (val: α) -> (err: String) -> InteractM α


inductive OpValid: OpVerifier -> Op -> Prop where
| MinArgs: ∀ (o: Op) (n: Nat) (PRF: (Op.args o).length >= n),
    OpValid (OpVerifier.MinArgs n) o
| MaxArgs: ∀ (o: Op) (n: Nat) (PRF: (Op.args o).length <= n),
    OpValid (OpVerifier.MaxArgs n) o
| ExactArgs: ∀ (o: Op) (n: Nat) (PRF: (Op.args o).length = n),
    OpValid (OpVerifier.ExactArgs n) o
-- | And: ∀ (op: Op) (l: OpVerifier) (r: OpVerifier)
--     (PRFL: OpValid l op) (PRFR: OpValid r op), OpValid (OpVerifier.And l r) op
-- | MinRegions: ∀ (o: Op) (n: Nat) (PRF: (Op.regions o).length >= n),
--    OpValid (OpVerifier.MinRegions n) o
| True: ∀ (o: Op), OpValid OpVerifier.T o


@[simp, reducible]
def OpVerifier.run (v: OpVerifier) (o: Op): InteractM Op :=
match v with
| OpVerifier.T => InteractM.ok o
| OpVerifier.MinArgs n => 
  if (Op.args o).length >= n
  then InteractM.ok  o
  else InteractM.error o ("expected >= " ++ toString n ++ "args")
| OpVerifier.MaxArgs n => 
   if (Op.args o).length <= n
   then InteractM.ok o
   else InteractM.error o ("expected <=" ++ toString n ++ "args") 
| OpVerifier.ExactArgs n => 
   if (Op.args o).length = n
   then InteractM.ok o
   else InteractM.error o ("expected ==" ++ toString n ++ "args")
-- | OpVerifier.MinRegions n => (Op.regions o).length >= n


/-
-- | reflection principle for opValid
-- | TODO: need someone who has done proofs in lean.
-- https://leanprover.github.io/lean4/doc/tactics.html
-- TODO: figure out which library theorem does this
theorem and_true__lhs_true (a b : Bool) : (a && b) = true → a = true := by {
  induction a;
  simp;
  simp;
}
-- TODO: figure out which library theorem does this
theorem and_true__rhs_true (a b : Bool) : (a && b) = true → b = true := by {
  induction a;
  simp;
  simp;
  intro h;
  exact h;
}

theorem opValid_implies_OpValid:
  ∀ (o : Op) (v: OpVerifier), (v.run o = True) → OpValid v o :=  by {
    intros o v h;
    induction v with
    | MinArgs n => exact (OpValid.MinArgs o n (of_decide_eq_true h));
    | MaxArgs n => exact (OpValid.MaxArgs o n (of_decide_eq_true h));
    | ExactArgs n => exact (OpValid.ExactArgs o n (of_decide_eq_true h));
    | And lhs rhs lhsValid rhsValid =>
      simp at h;
      apply OpValid.And; {
        apply lhsValid;
        apply (and_true__lhs_true _ _ h);
      } {
        apply rhsValid;
        apply (and_true__rhs_true _ _ h);
      }
    | MinRegions n => exact (OpValid.MinRegions _ _ (of_decide_eq_true h));
    | T => exact (OpValid.True _);
}

theorem OpValid_implies_opValid:
  ∀ (o : Op) (v: OpVerifier), OpValid v o → (v.run o = True) :=  by {
    intros o v h;
    induction h with
    | MinArgs o n prf => exact (decide_eq_true prf);
    | MaxArgs o n prf => exact (decide_eq_true prf);
    | ExactArgs o n prf => exact (decide_eq_true prf);
    | MinRegions o n prf => exact (decide_eq_true prf);
    | True => simp;
    | And op lhs rhs _ _ hlhs hrhs => simp; rewrite [hlhs, hrhs]; simp;
}
-/

theorem opValid_implies_OpValid:
  ∀ (o : Op) (v: OpVerifier), (v.run o = InteractM.ok o) → OpValid v o :=  sorry

theorem OpValid_implies_opValid:
  ∀ (o : Op) (v: OpVerifier), OpValid v o → (v.run o = InteractM.ok o) :=  sorry

theorem reflect_OpValid_opValid (o : Op) (v: OpVerifier) :
  OpValid v o ↔ (v.run o = InteractM.ok o) := 
   ⟨OpValid_implies_opValid o v, opValid_implies_OpValid o v⟩



@[simp]
class DialectOps (Ops : Type) where
  enumerate : List Ops
  verifier : Ops -> OpVerifier



-- def Region.recursivelyVerify (r: Region) (DO: Type) [Coe DO String] [DialectOps DO]: Bool := 


-- def Op.recursivelyVerify (op: Op) (DO: Type) [Coe DO String] [DialectOps DO]: Bool :=
--   let opKinds : List DO := DialectOps.enumerate 
--   let valid := match opKinds.find? (fun k => coe k == op.name) with
--                | some k =>  (DialectOps.verifier k).run op
--                | none => true
--   op.regions.all (fun r => r.recursivelyVerify DO)
-- 
-- end

-- | proof carrying typed operations
structure OpT (Ops: Type) [Coe Ops String] [DialectOps Ops]: Type where
    (kind: Ops)
    (args: List SSAVal)
    (bbs: List BBName)
    (regions: List Region) 
    (attrs: AttrDict)
    (ty: MLIRTy)
    (VALID: OpValid (DialectOps.verifier kind) (Op.mk kind args bbs regions attrs ty))



inductive DiffOps where
| var | add | mul | d

instance : Coe DiffOps String where
  coe: DiffOps -> String
  | DiffOps.var => "var"
  | DiffOps.add => "add"
  | DiffOps.mul => "mul"
  | DiffOps.d => "d"

@[simp,reducible]
instance : DialectOps DiffOps where
  enumerate := [DiffOps.var, DiffOps.add, DiffOps.mul, DiffOps.d]
  verifier : DiffOps -> OpVerifier
  | DiffOps.var => OpVerifier.ExactArgs 0
  | DiffOps.add => OpVerifier.ExactArgs 2
  | _ => OpVerifier.T


-- | diff operation, with num. args checked.
def varOpTRaw : InteractM Op :=
    (DialectOps.verifier DiffOps.var).run [mlir_op| "var"(%x) : i32]
-- #reduce varOpTRaw
 

 syntax ident "(" mlir_op_operand,* ")" 
  ("[" mlir_op_successor_arg,* "]")? ("(" mlir_region,* ")")?  ("{" mlir_attr_entry,* "}")? ":" mlir_type : mlir_op

macro_rules 
  | `([mlir_op| $x:ident 
        ( $operands,* )
        $[ [ $succ,* ] ]?
        $[ ( $rgns,* ) ]?
        $[ { $attrs,* } ]? : $ty:mlir_type ]) => do
        let initList <- `([])
        let operandsList <- operands.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_op_operand| $x]])
        let succList <- match succ with
                | none => `([])
                | some xs => xs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_op_successor_arg| $x] ])
        let attrsList <- match attrs with 
                          | none => `([]) 
                          | some attrs => attrs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_attr_entry| $x]])
        let rgnsList <- match rgns with 
                          | none => `([]) 
                          | some rgns => rgns.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_region| $x]])
        `({ kind := $x -- name
            , args := $operandsList -- operands
            , bbs := $succList -- bbs
            , regions := $rgnsList -- regions
            , attrs := (AttrDict.mk $attrsList) -- attrs
            , ty := [mlir_type| $ty]
            , VALID := opValid_implies_OpValid _ _ rfl })

def varOpTPretty : OpT DiffOps := 
  [mlir_op| DiffOps.var () : i32]
#check varOpTPretty




inductive RewriterT (Ops: Type) where
| op: (ty: Ops) -> (args: List SSAVal) -> (SSAVal -> RewriterT Ops) -> RewriterT Ops
| result: (op: SSAVal) -> (ix: Nat) -> (SSAVal -> RewriterT  Ops) -> RewriterT Ops
| operand:  (SSAVal -> RewriterT Ops) -> RewriterT Ops -- fresh operand
| rewrite: RewriterT Ops -> RewriterT Ops -- begin a rewrite block
| replace: (old: SSAVal) -> (new: SSAVal) 
   -> RewriterT Ops -> RewriterT Ops
| done: RewriterT Ops 



-- | d(x, add(p, q)) = add(d(x, p), d(x, q))
def DiffOpsPushAddRewriter: RewriterT DiffOps  := 
  RewriterT.op DiffOps.var [] $ λ x => 
  RewriterT.op DiffOps.var [] $ λ p => 
  RewriterT.op DiffOps.var [] $ λ q => 
  RewriterT.op DiffOps.add [p, q] $ λ add =>
  RewriterT.op DiffOps.d [x, add] $ λ diff =>
  RewriterT.rewrite $ 
  RewriterT.op DiffOps.d [x, p] $ λ dxp =>
  RewriterT.op DiffOps.d [x, q] $ λ dxq =>
  RewriterT.op DiffOps.add [dxp, dxq] $ λ add' => 
  RewriterT.replace add add' $
  RewriterT.done
  

-- | pattern match on the left
-- | TODO: is there some convenient way to avoid the final id lens?
def lensPushAddLHS : List (OpLens (ULift SSAVal)) := 
  [OpLens.arg 0 (ValLens.id), 
   OpLens.arg 1 (ValLens.op "add" (OpLens.arg 0 ValLens.id)),
   OpLens.arg 1 (ValLens.op "add" (OpLens.arg 1 ValLens.id))]

inductive interactive: (a: Type k) -> a -> a -> Type (k+1) where
| root: (v: a) -> interactive a v  v
| layer: (f: a -> a) -> (r: interactive a (f w) v) -> interactive a w (f v)
/-
-- | a proxy to show the value v
inductive Proxy {t: Type} (v: t) where
| proxy: Proxy v

  --  (RewriterT Ops Verifier) -> Type where
| op: (ty: Ops) -> (args: List SSAVal) -> 
      (rest: @Proxy (RewriterT Ops) (RewriterT.op ty args (fun _ => RewriterT.done))
             -> SSAVal
             -> RewriterTBuilderProxy Ops Verifier)
      -> RewriterTBuilderProxy Ops Verifier

-/

-- | using framework
/-
def DiffOpsPushAddProxy: RewriterTBuilderProxy DiffOps DiffVerifier := by {
  apply RewriterTBuilderProxy.op DiffOps.add;
  intros cur;
  intros x;
  apply RewriterTBuilderProxy.op DiffOps.add;
  intros cur; intros y;
  sorry;
}


inductive RewriterTBuilderIndexed (Ops: Type) (Verifier: Ops -> OpVerifier):
  RewriterT Ops Verifier -> Type
| op: (op: Ops)
      -> (args: List SSAVal)
      -> (rest: SSAVal -> RewriterTBuilderIndexed
      -> RewriterTBuilderIndexed Ops Verifier RewriterT.done
               
 -- | using framework
def DiffOpsPushAddIndexed: Σ r, RewriterTBuilderIndexed DiffOps DiffVerifier r  := by {
  apply RewriterTBuilderIndexed.op DiffOps.add [];
  intros x;
  apply RewriterTBuilderIndexed.op DiffOps.add [];
  intros y;
  sorry;
}
-/
                                 
def main_end_to_end_diff: IO Unit := do
  IO.eprintln "DIFF TEST\n=======\n"
  -- IO.eprintln matmul_ein
  -- IO.eprintln matmul_linalg
  -- IO.eprintln rewrite_antisym_sum
  -- IO.eprintln pdl_rewrite_antisym_sum
