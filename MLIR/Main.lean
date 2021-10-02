-- main entrypoint
import MLIR.P
import MLIR.Doc
import MLIR.AST
import MLIR.MLIRParser
import MLIR.EDSL
import Init.System.IO

open MLIR.MLIRParser
open MLIR.P
open MLIR.Doc
open MLIR.AST
open IO
open System


-- TOPLEVEL PARSER
-- ==============

-- https://github.com/leanprover/lean4/blob/master/tests/playground/file.lean
def main (xs: List String): IO Unit := do
  -- let path : System.FilePath :=  xs.head!
  let path :=  xs.head!
  let contents ← FS.readFile path;
  IO.eprintln "FILE\n====\n"
  IO.eprintln contents
  -- IO.eprintln "\nEDSL TESTING\n============\n"
  -- IO.eprintln MLIR.EDSL.opRgnAttr0
  IO.eprintln "PARSING\n=======\n"
  let ns := []
  let (loc, ns, _, res) <-  (pop ()).runP locbegin ns contents
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
     
  return ()
