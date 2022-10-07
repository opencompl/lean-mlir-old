/-
Define the semantics of core Linalg operations.
-/
import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
open MLIR.AST

instance linalg: Dialect Void Void (fun x => Unit) where
  name := "linalg"
  iα := inferInstance
  iε := inferInstance



def OpM.findIndex (d: AttrDict δ) (key: String): OpM Δ Nat :=
 match d.find_nat key with
 | .some v => return v
 | _ => OpM.Error s!"{d}.lookup {key} failed to find int"

-- in general, xtract slice has offset, size, stride.
-- We ignore the stride and offset for now, just use size.
def linalg_semantics_op: IOp Δ → OpM Δ (TypedArgs Δ)
 | IOp.mk "linalg.extractslice1d" _ [⟨.tensor1d, t⟩] [] dict => do 
    let len ←  OpM.findIndex dict "len"
    let t' := t.extract len
    return [⟨.tensor1d, t'⟩] 
 | IOp.mk "linalg.fill1d" _ [⟨.tensor1d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.generic1d" _ [⟨.tensor1d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.extractslice2d" _ [⟨.tensor1d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.insertslice2d" _ [⟨.tensor1d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.fill2d" _ [⟨.tensor1d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.generic2d" _ [⟨.tensor1d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.tile1d" _ [⟨.tensor1d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.transpose2d"   _ [⟨.tensor1d, t⟩]  [r] dict => sorry
 | IOp.mk "linalg.parallel2d" _ [⟨.tensor1d, t⟩]  [r] dict => do
      return []
 | _ => OpM.Unhandled "unhandled linalg.generic"

/-
For each transformation, we implement
1) a theorem that proves correctness
2) a test in Test/SemanticTests.lean which tests
   both versions of the program.
-/
namespace BubbleUpExtractSlice
-- https://mlir.llvm.org/doxygen/BubbleUpExtractSlice_8cpp_source.html
def extract_slice_fill_1d : Bool := true
def extract_slice_fill_2d : Bool := true
def extract_slice_generic_2d : Bool := true
end BubbleUpExtractSlice
