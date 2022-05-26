/-
## IRDL

This file IRDL structures that are used to represent dialects.
-/

import MLIR.AST
open MLIR.AST

inductive IRDLTypeConstraint (δ: Dialect α σ ε) :=
  | TypeVar (typeName: String)
  | Eq (τ: MLIRType δ)
  | AnyOf (constraint: List (IRDLTypeConstraint δ))

def IRDLTypeConstraint.verify {δ: Dialect α σ ε} (constr: IRDLTypeConstraint δ) (τ: MLIRType δ) : Bool :=
  match constr with
  | TypeVar name => true
  | Eq τ' => τ == τ'
  | AnyOf [] => false
  | AnyOf (constr::constrs) => constr.verify τ || ((AnyOf constrs).verify τ)

structure IRDLOp {δ: Dialect α σ ε} where mk ::
  -- Operation name
  name: String
  -- Constraint variables
  constraint_vars: List (String × IRDLTypeConstraint δ) := []
  -- Operand type constraints
  operands: List (String × IRDLTypeConstraint δ) := []
  -- Result type constraints
  results: List (String × IRDLTypeConstraint δ) := []

