import Lean.Parser
import Lean.Parser.Extra

open Lean
open Lean.Parser

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
| BasicBlock: (ops: List Op) -> BasicBlock

inductive Region: Type where
| Region: (bbs: List BasicBlock) -> Region
end



notation "{" "}" => Region.Region  []
notation:65 "module" r => Op.MkOp "module" [] [] r

def empty_module : Op := module { }

