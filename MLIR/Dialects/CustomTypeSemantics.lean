/-
This dialect tests the dialect extension capability to define
custom types
-/
import MLIR.Semantics.Fitree
import MLIR.Semantics.Semantics
import MLIR.Semantics.SSAEnv
import MLIR.Semantics.UB
import MLIR.Util.Metagen
import MLIR.AST
import MLIR.EDSL
open MLIR.AST


inductive CustomType where
| CBool
deriving DecidableEq

@[reducible]
def CustomType.ε: CustomType -> Type
| .CBool => Bool

instance : DialectTypeIntf CustomType CustomType.ε where
  inhabited := by {
      intros s; cases s;simp [CustomType.ε]; exact true;
  }
  typeEq := inferInstance
  eq := by { intros x; simp; cases x; exact inferInstance; }
  str := fun _ _ => "Bool"
  typeStr := fun _ => "Bool"

instance CustomTypeDialect: Dialect Void CustomType CustomType.ε where
  name := "customtype"
  iα := inferInstance
  iε := inferInstance


#check MLIRType.eval
def customTypeDialectSemanticsOp: 
  IOp CustomTypeDialect → OpM CustomTypeDialect (TypedArgs CustomTypeDialect)
  | IOp.mk "customtype.true" _ [] [] _ => 
    return [⟨ .extended .CBool, true⟩]
  | IOp.mk "customtype.if" _ [⟨.extended .CBool, b⟩]  [rthen, relse] _ => do
      if b then rthen [] else relse []
  | _ => OpM.Unhandled s!"unhandled op"

instance: Semantics CustomTypeDialect where
  semantics_op := customTypeDialectSemanticsOp

