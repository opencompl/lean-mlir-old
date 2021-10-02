-- Write a parser imperatively using the Substring mechanism.

import Init.Data.String
import Init.Data.String.Basic
import Init.System.IO

open String
open Substring
open Lean
open IO
open System

inductive Result (e : Type) (a : Type) : Type where 
| ok: a -> Result e a
| err: e -> Result e a

instance [Inhabited e] : Inhabited (Result e a) where
   default := Result.err (Inhabited.default) 


structure Loc where
  line : Int
  column : Int

instance : Inhabited Loc where
   default := { line := 1, column := 1 }


instance : ToString Loc := {
  toString := fun loc => 
    toString loc.line ++ ":" ++ toString loc.column
}


def locbegin : Loc := { line := 1, column := 1 }

-- | move a loc by a string.
def advance (l: Loc) (s: Substring): Loc :=
  if s.isEmpty then l
  else if front s == '\n'
    then { line := l.line + 1, column := 1  }
    else { line := l.line, column := l.column + 1}
 
def advance1 (l: Loc) (c: Char): Loc :=
  if c == '\n'
    then { line := l.line + 1, column := 1  }
    else return { line := l.line, column := l.column + 1}


-- https://github.com/leanprover/lean4/blob/master/tests/playground/file.lean
def main (xs: List String): IO Unit := do
  -- let path : System.FilePath :=  xs.head!
  let path :=  xs.head!
  let contents ← FS.readFile path;
  IO.println "FILE\n====\n"
  IO.println contents
  IO.println "PARSING\n=======\n"
  -- let res := parse_op locbegin contents
  -- match res with
  --  | Result.ok (loc, str, op) => IO.println op
  --  | Result.err res => IO.println res
  -- return ()

