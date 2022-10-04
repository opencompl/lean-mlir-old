import MLIR.Doc
import MLIR.CParser
import MLIR.FFI
import MLIR.AST
import MLIR.P
import MLIR.EDSL
import MLIR.Tests.TestLib
import MLIR.Tests.AllTests
import MLIR.Examples.FinIntBruteforce

-- Testing imports for semantics
import MLIR.Dialects.ToySemantics
import MLIR.Dialects.PDLSemantics
-- import MLIR.Dialects.ControlFlowSemantics
import MLIR.Dialects.ScfSemantics
import MLIR.Dialects.ArithSemantics

open MLIR.P
open MLIR.Doc
open MLIR.AST
open IO
open System


-- TOPLEVEL PARSER
-- ==============

-- https://github.com/leanprover/lean4/blob/master/tests/playground/file.lean
def main (xs: List String): IO UInt32 := do
  if xs.length == 0 then
    IO.println "usage: mlir [--run-test-suite] [--extract-semantic-tests]"
    return 0
  else if xs == ["--run-test-suite"] then
    let b ← TestLib.runTestSuite AllTests.testSuite
    return (if b then 0 else 1)
  else if xs.length == 2 && xs.head! == "--extract-semantic-tests" then
    SemanticsTests.semanticTests.forM fun t => do
      let (SemanticsTests.SemanticTest.mk S name r) := t
      let fn: Op (S + scf) := .mk "func.func" [] [] [r]
        (.mk [.mk "sym_name" $ .str "main"])
      let rgn: Region (S + scf) := .mk "entry" [] [fn]
      let m: Op (S + scf) := .mk "builtin.module" [] [] [rgn] .empty
      let out_folder := xs.drop 1 |>.head!
      IO.println s!"extracting {out_folder}/{name}..."
      let code := layout80col $ Pretty.doc m
      FS.writeFile s!"{out_folder}/{name}" code
    return 0
  else
    return 0
