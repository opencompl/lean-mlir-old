import MLIR.AST
import MLIR.EDSL
import MLIR.PDL

open MLIR.AST

inductive OpVerifier where
| MinArgs: Nat -> OpVerifier
| MaxArgs: Nat -> OpVerifier
| ExactArgs: Nat -> OpVerifier
| And: OpVerifier -> OpVerifier -> OpVerifier
| MinRegions: Nat -> OpVerifier
| T: OpVerifier


inductive OpValid: OpVerifier -> Op -> Prop where
| MinArgs: ∀ (o: Op) (n: Nat) (PRF: (Op.args o).length >= n),  OpValid (OpVerifier.MinArgs n) o
| MaxArgs: ∀ (o: Op) (n: Nat) (PRF: (Op.args o).length <= n),  OpValid (OpVerifier.MaxArgs n) o
| ExactArgs: ∀ (o: Op) (n: Nat) (PRF: (Op.args o).length == n),  OpValid (OpVerifier.ExactArgs n) o
| And: ∀ (op: Op) (l: OpVerifier) (r: OpVerifier)
    (PRFL: OpValid l op) (PRFR: OpValid r op), OpValid (OpVerifier.And l r) op
| True: ∀ (o: Op), OpValid OpVerifier.T o


def OpVerifier.run (v: OpVerifier) (o: Op): Bool :=
match v with
| OpVerifier.T => true
| OpVerifier.MinArgs n => (Op.args o).length >= n
| OpVerifier.MaxArgs n => (Op.args o).length <=  n
| OpVerifier.ExactArgs n => (Op.args o).length == n
| OpVerifier.And v v' => v.run o && v'.run o
| OpVerifier.MinRegions n => (Op.regions o).length >= n


-- | reflection principle for opValid
-- | TODO: need someone who has done proofs in lean.
theorem opValid_implies_OpValid: ∀ (v: OpVerifier) (o: Op)(VALID: v.run o = True), OpValid v o := 
   sorry

@[simp]
class DialectOps (Ops : Type) where
  enumerate : List Ops
  verifier : Ops -> OpVerifier

mutual

def Region.recursivelyVerify (r: Region) (DO: Type) [Coe DO String] [DialectOps DO]: Bool := 


def Op.recursivelyVerify (op: Op) (DO: Type) [Coe DO String] [DialectOps DO]: Bool :=
  let opKinds : List DO := DialectOps.enumerate 
  let valid := match opKinds.find? (fun k => coe k == op.name) with
               | some k =>  (DialectOps.verifier k).run op
               | none => true
  op.regions.all (fun r => r.recursivelyVerify DO)

end

-- | proof carrying typed operations
structure OpT (Ops: Type) [Coe Ops String] [DialectOps Ops]: Type where
    (kind: Ops)
    (args: List SSAVal)
    (bbs: List BBName)
    (regions: List Region) 
    (attrs: AttrDict)
    (ty: MLIRTy)
    (VALID: OpValid (DialectOps.verifier kind) (Op.mk kind args bbs regions attrs ty))


syntax ident  "(" mlir_op_operand,* ")" 
  ("[" mlir_op_successor_arg,* "]")? ("(" mlir_region,* ")")?  ("{" mlir_attr_entry,* "}")? ":" mlir_type : mlir_op

macro_rules
  | `([mlir_op|  $opty:ident
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
        `(OpT.mk $opty
                $operandsList -- operands
                $succList -- bbs
                $rgnsList -- regions
                (AttrDict.mk $attrsList) -- attrs
                [mlir_type| $ty]) -- type



inductive DiffOps where
| var | add | mul | d



instance : Coe DiffOps String where
  coe: DiffOps -> String
  | DiffOps.var => "var"
  | DiffOps.add => "add"
  | DiffOps.mul => "mul"
  | DiffOps.d => "d"

@[simp]
instance : DialectOps DiffOps where
  enumerate := [DiffOps.var, DiffOps.add, DiffOps.mul, DiffOps.d]
  verifier : DiffOps -> OpVerifier
  | DiffOps.var => OpVerifier.ExactArgs 0
  | DiffOps.add => OpVerifier.ExactArgs 2
  | _ => OpVerifier.T



-- | diff operation, with num. args checked.
def addOpT : OpT DiffOps := 
   let VALID : _ := by {  constructor; simp; }
   [mlir_op| DiffOps.add (%x, %y) : i32] VALID

#check addOpT




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
