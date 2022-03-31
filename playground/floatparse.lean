import Lean.Parser 
open Lean
open Lean.Parser

inductive  AST
| int: Int -> AST
| float: Float -> AST



declare_syntax_cat foo
syntax  numLit : foo
syntax "[foo|" foo "]" : term

set_option maxRecDepth 999999

macro_rules
| `([foo| $x:numLit]) => `(AST.int $x)

def val0 : AST := [foo| 0 ]
#reduce val0

syntax scientificLit : foo
macro_rules
| `([foo| $x:scientificLit]) => `(AST.float $x)
def val42 : AST := [foo| 42.34 ]
#print val42

def valMLIR : AST := [foo| 0.000000e00  ]
#print valMLIR

-- def attrVal10Float : AttrVal := [mlir_attr_val| 0.0023 ]

