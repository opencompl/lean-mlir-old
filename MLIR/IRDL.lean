/-
## IRDL

This file IRDL structures that are used to represent dialects.
-/

import MLIR.AST
import MLIR.Semantics.Types
open MLIR.AST

-- Type constraints understood by IRDL
inductive IRDLTypeConstraint (δ: Dialect α σ ε) :=
  /- A type constraint variable, all type constraint variables of
     the same name should be satisfied by the same type. Each type
     constraint variable also has an associated type constraint. -/
  | TypeVar (typeName: String)
  | Eq (τ: MLIRType δ) -- An equality type constraint
  /- A type constraint that is satisfied by types satisfying any of
     the given type constraints. -/
  | AnyOf (constraint: List (IRDLTypeConstraint δ))
  deriving Inhabited

@[reducible]
abbrev ConstraintMapping (δ: Dialect α σ ε) := List (String × IRDLTypeConstraint δ × Option (MLIRType δ))

-- Check that a type satisfy the invariants specified by a type constraint
-- `constrVars` contains the constraints specified by the type constraint variables.
-- `constrVars` constraints should only refer to constraint variables in smaller indices.
def IRDLTypeConstraint.verify_ {δ: Dialect α σ ε} (constr: IRDLTypeConstraint δ) 
                              (constrVars: ConstraintMapping δ)
                              (τ: MLIRType δ): Option (ConstraintMapping δ) :=
  -- Lean requires us to write it this way to prove termination.
  let rec go (constr: IRDLTypeConstraint δ) (constrVars: ConstraintMapping δ) := 
    match constr with
    -- For TypeVars, we get the associated constraint
    | TypeVar name =>
      match constrVars with
      | [] => panic "Can't find any constraint with that name"
      | (name', (constr', val))::constrVars' =>
        -- When we find the constraint
        if name == name' then 
          match val with
          -- If the constraint variable has already an associated type, we check that the type is equal.
          | some τ' => if τ == τ' then some constrVars else none
          -- Otherwise, we check that the type satisfies the variable, and we assign the type to the varible.
          | none => (go constr' constrVars').map (fun constrVars'' => ((name', (constr', some τ))::constrVars''))
        -- Otherwise, we look for the constraint recursively
        else
          (go constr constrVars').map (fun constrVars'' => ((name', (constr, val))::constrVars''))
    -- For the equality case, we just check for equality.
    | Eq τ' => if τ == τ' then some constrVars else none
    -- For the union case, we take the first constraint that is satisfied.
    -- Note that this mean that the algorithm is not complete, since the assignment of
    -- the type constraint variable might not be "optimal".
    | AnyOf [] => none
    | AnyOf (constr::constrs) =>
      match go constr constrVars with
      | some constrVars => some constrVars
      | none => go (AnyOf constrs) constrVars
go constr constrVars
termination_by go constr constrVars =>  (constrVars, constr) 
  
-- An IRDL definition of an operation.
structure IRDLOp {δ: Dialect α σ ε} where mk ::
  -- Operation name
  name: String
  -- Constraint variables
  constraint_vars: List (String × IRDLTypeConstraint δ) := []
  -- Operand type constraints
  operands: List (String × IRDLTypeConstraint δ) := []
  -- Result type constraints
  results: List (String × IRDLTypeConstraint δ) := []

