-- Trying to reproduce a bug with $(..) within a large syntax object
import Lean.Parser
import Lean.Parser.Extra

open Lean
open Lean.Parser

-- | TODO: factor Symbol out from AttrVal
inductive AttrVal : Type where
| str : String -> AttrVal

inductive AttrEntry : Type where
  | mk: (key: String) 
      -> (value: AttrVal)
      -> AttrEntry

declare_syntax_cat mlir_attr_val
syntax str: mlir_attr_val
syntax "[mlir_attr_val|" incQuotDepth(mlir_attr_val) "]" : term

macro_rules
| `([mlir_attr_val| $$($x) ]) => `($x)
| `([mlir_attr_val| $s:strLit]) => `(AttrVal.str $s)

-- Attribute Entries
declare_syntax_cat mlir_attr_entry

syntax strLit "=" mlir_attr_val : mlir_attr_entry
syntax "[mlir_attr_entry|" incQuotDepth(mlir_attr_entry) "]" : term

macro_rules 
  | `([mlir_attr_entry| $name:strLit  = $v:mlir_attr_val]) => 
     `(AttrEntry.mk $name [mlir_attr_val| $v])

def attrVal0Str : AttrVal := [mlir_attr_val| "add"]
#reduce attrVal0Str

def attrVal1Escape : AttrVal := [mlir_attr_val| $(attrVal0Str)]
#reduce attrVal1Escape

def AttrEntry0Str : AttrEntry := [mlir_attr_entry| "sym_name" = "add"]
#reduce AttrEntry0Str


-- vv example SHOULD NOT FAIL?
def AttrEntry1Escape : AttrEntry := [mlir_attr_entry| "sym_name" = $(attrVal0Str)]
#reduce AttrEntry1Escape



declare_syntax_cat inner
declare_syntax_cat outer

syntax "<[" inner "]>" : outer

syntax "[inner|" incQuotDepth(inner) "]" : term
macro_rules
| `([inner| $$($s)]) => return s

syntax "[outer|" outer "]" : term
macro_rules
| `([outer| <[ $k:inner ]> ]) => `([inner| $k])

syntax "[outerIncQuotDepth|" incQuotDepth(outer) "]" : term
macro_rules
| `([outerIncQuotDepth|  <[ $k:inner ]> ]) => `([inner| $k])


def g := "global string"

def innerQuot := [inner| $(g) ]
def outerIncQuotDepth := [outerIncQuotDepth| <[ $(g) ]> ]
-- I would expect `outerDoesNotWork` to work, because the quoting comes from inner, not outer!
def outerDoesNotWork := [outer| <[ $(g) ]> ] 
