import MLIR.Doc
import MLIR.CParser
import MLIR.FFI
import MLIR.AST
import MLIR.P
import MLIR.MLIRParser
import MLIR.EDSL
import MLIR.PatternMatch
import MLIR.Dialects.Builtin
import MLIR.Dialects.Linalg
import MLIR.Dialects.PDL
import MLIR.Examples.EndToEndComplex
import MLIR.Examples.EndToEndLinalg
import MLIR.Examples.EndToEndLz
import MLIR.Examples.EndToEndDiff
import MLIR.Examples.SemanticsTests
import MLIR.Examples.FinIntBruteforce

-- Testing imports for semantics
import MLIR.Dialects.ToySemantics
import MLIR.Dialects.PDLSemantics
import MLIR.Dialects.ControlFlowSemantics
import MLIR.Dialects.ArithSemantics

open MLIR.MLIRParser
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
    -- main_end_to_end_linalg
    main_end_to_end_lz
    main_end_to_end_diff
    return 0
  else if xs == ["--run-semantic-tests"] then
    let b ← SemanticsTests.runAllTests
    return (if b then 0 else 1)
  else if xs.length == 2 && xs.head! == "--extract-semantic-tests" then
    SemanticsTests.allTests.forM fun t => do
      let (SemanticsTests.Test.mk S name r) := t
      let τv: MLIRType (S + cf) := .fn (.tuple []) (.tuple [])
      let τi: MLIRType (S + cf) := .fn (.tuple []) (.tuple [.i32])
      let fn: Op (S + cf) := .mk "func.func" [] [] [r]
        (.mk [.mk "sym_name" $ .str "main",
              .mk "function_type" $ .type τi]) τv
      let bb: BasicBlock (S + cf) := .mk "entry" [] [.StmtOp fn]
      let m: Op (S + cf) := .mk "builtin.module" [] [] [.mk [bb]] .empty τv
      let out_folder := xs.drop 1 |>.head!
      IO.println s!"extracting {out_folder}/{name}..."
      let code := layout80col $ Pretty.doc m
      FS.writeFile s!"{out_folder}/{name}" code
    return 0
  else
    -- let path : System.FilePath :=  xs.head!
    let path :=  xs.head!
    let contents ← FS.readFile path;
    IO.eprintln "FILE\n====\n"
    IO.eprintln contents
    -- IO.eprintln "\nEDSL TESTING\n============\n"
    -- IO.eprintln MLIR.EDSL.opRgnAttr0
    IO.eprintln "PARSING\n=======\n"
    let ns := []
    let (loc, ns, _, res) := (pop (δ := builtin) ()).runP locbegin ns contents
    IO.eprintln (vgroup $ ns.map (note_add_file_content contents))
    match res with
     | Result.ok op => do
       IO.println op
     | Result.err err => do
        IO.eprintln "***Parse Error:***"
        IO.eprintln (note_add_file_content contents err)
     | Result.debugfail err =>  do
        IO.eprintln "***Debug Error:***"
        IO.eprintln (note_add_file_content contents err)
    return 0
