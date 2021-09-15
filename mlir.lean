import Lean.Parser
import Lean.Parser.Extra

open Lean
open Lean.Parser

structure SSAVal where
  name: String

structure Inst where
    name: String

structure Attribute where
    key: String
    value: String


structure Op where
    name : String
    args: List SSAVal
    attributes: List Attribute
    regions: List Region
    ty : MLIRType


structure Binding where
    lhs: SSAVal
    Op: Op

structure BasicBlock where
  name: String
  args: List SSAVal
  ops : List Binding


structure Region where
    blocks: List BasicBlock

