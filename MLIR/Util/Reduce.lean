/-
## Extended `#reduce` command

This command extends `#reduce` with a `(skipProofs := true/false)` parameter to
control whether proofs are reduced in the kernel. Reducing proofs often cause
timeouts, and they are used implicitly through well-founded induction for
mutual definitions.

See: https://leanprover.zulipchat.com/#narrow/stream/270676-lean4/topic/Repr.20instance.20for.20functions/near/276504682

Note that `MLIR/AST.lean` contains mutually-recursive dialect coercions for
`Op`/`Region`/etc which appear to time out even with `(skipProofs := true)`.
-/

import Lean

open Lean
open Lean.Parser.Term
open Lean.Elab.Command
open Lean.Elab
open Lean.Meta

elab "#reduce " skipProofs:group(atomic("(" &"skipProofs") " := " (trueVal <|> falseVal) ")") term:term : command =>
  let skipProofs := skipProofs.raw[3].isOfKind ``trueVal
  withoutModifyingEnv <| runTermElabM (some `_check) fun _ => do
    -- dbg_trace term
    let e ← Term.elabTerm term none
    Term.synthesizeSyntheticMVarsNoPostponing
    let (e, _) ← Term.levelMVarToParam (← instantiateMVars e)
    withTheReader Core.Context (fun ctx => { ctx with options := ctx.options.setBool `smartUnfolding false }) do
      let e ← withTransparency (mode := TransparencyMode.all) <| reduce e (skipProofs := skipProofs) (skipTypes := false)
      logInfo e
