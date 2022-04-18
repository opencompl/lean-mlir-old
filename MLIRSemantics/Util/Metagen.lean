import Lean.Parser
import Lean
open Lean.Elab.Command
open Lean.Elab.Term
open Lean.Meta
open Lean

-- Mock reproduction of MLIR.AST

private inductive SSAValue
| mk: String -> SSAValue

private def SSAValue.toString: SSAValue -> String
| SSAValue.mk a => a

private instance: ToString SSAValue where
  toString := SSAValue.toString

syntax "%" ident : term
macro_rules
| `(% $i:ident) => do
    let name := Lean.quote (i.getId.toString)
    `(SSAValue.mk $name)

private inductive Op
| Operand: Op
| Op_: String -> List SSAValue -> Op

private def Op.toString: Op → String
  | Operand => "operand"
  | Op_ name xs => s!"op {name} {xs}"

private instance: ToString Op where
  toString := Op.toString

inductive BB
| mk: List (SSAValue × Op) -> BB

def BB.toString: BB → String
  | BB.mk xs => "^entry:\n" ++ String.intercalate "\n" (
      (xs.map (fun (name, val) => s!"{name} := {val}")))

--


-- We want to generate commands based on data structures after reduction and
-- evaluation. But evalExpr is unsafe so elab_rules refuses it. We insist.
@[implementedBy evalExpr] -- HACK
def evalExprSafe (α) (typeName: Name) (value: Expr): TermElabM α :=
  throwError "trust me evalExpr is safe enough"

-- Variation of evalExpr that supports non-constant types.
-- | TODO: Find out how to reflect `α` into `expectedType`
unsafe def evalExprAnyType (α: Type) (expectedType: Expr) (value: Expr):
    TermElabM α :=
  withoutModifyingEnv do
    let name ← mkFreshUserName `_tmp
    let type ← inferType value
    synthesizeSyntheticMVarsNoPostponing
    unless (<- isDefEq type expectedType) do
      throwError "incorrect type {type}, expected type {expectedType}"

    let decl := Declaration.defnDecl {
       name := name, levelParams := [], type := type,
       value := value, hints := ReducibilityHints.opaque,
       safety := DefinitionSafety.unsafe
    }
    ensureNoUnassignedMVars decl
    addAndCompile decl
    evalConst α name

-- Same hack
@[implementedBy evalExprAnyType] -- HACK
def evalExprAnyTypeSafe (α) (expectedType: Expr) (value: Expr): TermElabM α :=
  throwError "trust me evalExprAnyType is safe enough"


--


set_option hygiene false in
elab "mkRewriteThm" name:ident x:term ":=" t:term : command => do
  let xbb <- liftTermElabM `mkRewriteThm do
    let x ← elabTerm x none
    let xred <- reduce x
    dbg_trace xred
    let xbb : BB <- evalExprSafe BB `BB xred
    return xbb
  let x <- `(def $name (argument: Nat): Nat := $t)
  elabCommand x



def bb0 := BB.mk
[
   (%v, Op.Operand),
   (%op, Op.Op_ "operation_name" [%v])
]

mkRewriteThm myAmazingRewrite bb0 := by {
  exact 42
}

#print myAmazingRewrite


--


structure OpSpec := mk ::
  name: String
  args: CommandElabM Syntax

elab "genInductive" inductiveName:ident xs:term : command => do
  let xargs ← liftTermElabM `genInductive do
    let xs ← elabTerm xs none
    let xsred ← instantiateMVars xs -- (← reduce xs)
    -- dbg_trace xsred
    let argType ← elabTerm (← `(Array OpSpec)) none
    let xsArray ← evalExprAnyTypeSafe (Array OpSpec) argType xsred
    -- dbg_trace xsArray
    return xsArray

  /- This will be useful later
  let make_ctor_signature (args: List String): CommandElabM Syntax := do
    let args := args.map (Lean.mkIdent ∘ Name.mkSimple)
    args.foldrM
      (fun name stx => `(($name :Type) → $name → $stx))
      (← `($inductiveName Nat))

  let xargs: Array (String × Syntax) ← xargs.mapM (fun spec => do
    return (spec.name, ← make_ctor_signature spec.args))
  -/

  let xargs: Array (String × Syntax) ← xargs.mapM (fun spec => do
      return (spec.name, ← spec.args))

  let make_ctor (spec: String × Syntax): CtorView :=
    let (name, stx) := spec
    let defaultCtorView : CtorView := default
    { defaultCtorView with
        declName := inductiveName.getId ++ name,
        type? := stx }

  let indView : InductiveView := {
    ref := Syntax.missing,
    modifiers := default,
    shortDeclName := inductiveName.getId,
    declName := inductiveName.getId,
    levelNames := [],
    binders := Syntax.missing,
    type? := some $ ← `(Type → Type 1),
    ctors := xargs.map make_ctor,
    derivingClasses := #[]
  }
  elabInductiveViews #[indView]


set_option hygiene false in
genInductive Test #[
  OpSpec.mk "Constant" `(Unit → Test Nat),
  OpSpec.mk "Transpose" `(Nat → Test Nat),
  OpSpec.mk "Reshape" `(Nat → Test Bool)
]
#print Test
