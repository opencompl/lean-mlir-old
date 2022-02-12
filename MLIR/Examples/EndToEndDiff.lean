import MLIR.AST

open MLIR.AST

inductive OpVerifier where
| MinArgs: Nat -> OpVerifier
| MaxArgs: Nat -> OpVerifier
| ExactArgs: Nat -> OpVerifier
| And: OpVerifier -> OpVerifier -> OpVerifier
| MinRegions: Nat -> OpVerifier
| T: OpVerifier


inductive OpValid: OpVerifier -> Op -> Prop where
| True: ∀ (o: Op), OpValid OpVerifier.T o


-- | proof carrying typed operations 94
structure OpT (Ops: Type) [Coe Ops String] (Verifier: Ops -> OpVerifier): Type where
    (kind: Ops) 
    (args: List SSAVal)
    (bbs: List BBName)
    (regions: List Region) 
    (attrs: AttrDict)
    (ty: MLIRTy)
    (VALID: OpValid (Verifier kind) (Op.mk kind args bbs regions attrs ty))

inductive DiffOps where
| var
| add
| mul
| sin
| d

def DiffVerifier: DiffOps -> OpVerifier
| DiffOps.var => OpVerifier.ExactArgs 1 
| DiffOps.add => OpVerifier.ExactArgs 2
| _ => OpVerifier.T



inductive RewriterT (Ops: Type) (Verifier: Ops -> OpVerifier)
 where
| op: (ty: Ops) -> (args: List SSAVal) -> (SSAVal -> RewriterT Ops Verifier) -> RewriterT Ops Verifier
| result: (op: SSAVal) -> (ix: Nat) -> (SSAVal -> RewriterT  Ops Verifier) -> RewriterT Ops Verifier
| operand:  (SSAVal -> RewriterT Ops Verifier) -> RewriterT Ops Verifier -- fresh operand
| rewrite: RewriterT Ops Verifier -> RewriterT Ops Verifier -- begin a rewrite block
| replace: (old: SSAVal) -> (new: SSAVal) 
   -> RewriterT Ops Verifier -> RewriterT Ops Verifier
| done: RewriterT Ops Verifier



-- | d(x, add(p, q)) = add(d(x, p), d(x, q))
def DiffOpsPushAdd0: RewriterT DiffOps DiffVerifier := 
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


-- | a proxy to show the value v
inductive Proxy {t: Type} (v: t) where
| proxy: Proxy v

inductive RewriterTBuilderProxy (Ops: Type) (Verifier: Ops -> OpVerifier): Type where
  --  (RewriterT Ops Verifier) -> Type where
| op: (ty: Ops) -> (args: List SSAVal) -> 
      (rest: @Proxy (RewriterT Ops Verifier) (RewriterT.op ty args (fun _ => RewriterT.done))
             -> SSAVal
             -> RewriterTBuilderProxy Ops Verifier)
      -> RewriterTBuilderProxy Ops Verifier


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
