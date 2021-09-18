import Lean.Parser
import Lean.Parser.Extra

open Lean
open Lean.Parser

-- EMBEDDING
-- ==========

mutual
inductive SSAVal : Type where
  | SSAVal : String -> SSAVal

inductive Attribute : Type where
  | Attribute: (key: String) 
      -> (value: String)
      -> Attribute

inductive Op : Type where 
 | MkOp: (name: String) 
      -> (args: List SSAVal)
      -> (attrs: List Attribute)
      -> (region: List Region) -> Op

inductive Path : Type where 
 | PathComponent: (regionix : Int) 
    -> (bbix: Int) 
    -> (opix: Int)
    -> (rec: Path)
    -> Path
 | Path

inductive BasicBlock: Type where
| BasicBlock: (name: String) -> (ops: List Op) -> BasicBlock

inductive Region: Type where
| Region: (bbs: List BasicBlock) -> Region
end


-- PARSER
-- ==========

inductive Result (e : Type) (a : Type) : Type where 
| ok: a -> Result e a
| err: e -> Result e a

inductive ParseErrorType : Type where
| ParseErrorGeneric : (name : String) -> ParseErrorType

structure Loc where
  line : Int
  column : Int

structure ParseError where
  left : Loc
  right : Loc
  error : ParseErrorType

def P (a: Type) : Type := String -> Result ParseError (String × a)

-- https://github.com/leanprover/lean4/blob/d0996fb9450dc37230adea9d10ecfdf10330ef67/tests/playground/flat_parser.lean
def ppure {a: Type} (v: a): P a := λs => Result.ok (s, v)

def pbind {a b: Type} (pa: P a) (a2pb : a -> P b): P b := 
   λs => match pa s with 
            | Result.ok (s', a) => a2pb a s'
            | Result.err e => Result.err e

instance : Monad P := {
  pure := ppure,
  bind := pbind
}

-- parseMLIRModule :: String -> Parser Module 

-- EDSL MACRO
-- ==========

notation:100 "{" "}" => Region.Region  []
notation:100 "{" bbs "}" => Region.Region bbs
notation:100 "module" r => Op.MkOp "module" [] [] [r]
-- notation:100 "^" name ":" => BasicBlock.BasicBlock name []
-- https://arxiv.org/pdf/2001.10490.pdf#page=11
macro "^" n:str ":" : term => `(BasicBlock.BasicBlock $n [])
macro "^" n:str ":" ops:term : term => `(BasicBlock.BasicBlock $n $ops)


def empty_module : Op := module { }
def empty_bb : BasicBlock := 
 ^"entry":  []


