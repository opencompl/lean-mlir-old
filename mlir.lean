import Lean.Parser
import Lean.Parser.Extra

open Lean
open Lean.Parser

mutual
-- | TODO: make this work with structure?
inductive SSAVal : Type where
  | SSAVal : String -> SSAVal


inductive Attribute : Type where
  | Attribute: (key: String) -> (value: String) -> Attribute

inductive Op : Type where 
 | Op: (name: String) -> (args: List SSAVal) -> (attrs: List Attribute) -> (region: List Region) -> Op

-- | A path to a particular value.
inductive Path : Type where 
 | PathComponent: (regionix : Int) -> (bbix: Int) -> (opix: Int) -> (rec: Path) -> Path
 | Path

inductive BasicBlock: Type where
| BasicBlock: (ops: List Op) -> BasicBlock

inductive Region: Type where
| Region: (bbs: List BasicBlock) -> Region
end
