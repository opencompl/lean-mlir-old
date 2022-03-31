-- Trying to reproduce a bug with $(..) within a large syntax object
import Lean.Parser
import Lean.Parser.Extra

open Lean
open Lean.Parser

-- | TODO: factor Symbol out from AttrVal
inductive AttrVal : Type where
| symbol: String -> AttrVal -- symbol ref attr
| str : String -> AttrVal
| int : Int -> AttrVal
| float : Float  -> AttrVal
-- | type :MLIRTy -> AttrVal
-- | dense: TensorElem -> MLIRTy -> AttrVal -- dense<10> : vector<i32>
-- | affine: AffineMap -> AttrVal
| list: List AttrVal -> AttrVal
-- | guaranteee: both components will be AttrVal.Symbol.
-- | TODO: factor symbols out.
| nestedsymbol: AttrVal -> AttrVal -> AttrVal 
| alias: String -> AttrVal
-- | dict: AttrDict -> AttrVal

-- https://mlir.llvm.org/docs/LangRef/#attributes
-- | TODO: add support for mutually inductive records / structures
inductive AttrEntry : Type where
  | mk: (key: String) 
      -> (value: AttrVal)
      -> AttrEntry

declare_syntax_cat mlir_attr_entry

declare_syntax_cat mlir_attr_val
declare_syntax_cat mlir_attr_val_symbol
syntax "@" ident : mlir_attr_val_symbol
syntax "@" str : mlir_attr_val_symbol
syntax "#" ident : mlir_attr_val -- alias
syntax "#" strLit : mlir_attr_val -- aliass
syntax mlir_attr_val_symbol "::" mlir_attr_val_symbol : mlir_attr_val_symbol


syntax str: mlir_attr_val
-- syntax mlir_type : mlir_attr_val
-- syntax affine_map : mlir_attr_val
syntax mlir_attr_val_symbol : mlir_attr_val
-- syntax num (":" mlir_type)? : mlir_attr_val
-- syntax scientificLit (":" mlir_type)? : mlir_attr_val

syntax "[" sepBy(mlir_attr_val, ",") "]" : mlir_attr_val
syntax "[mlir_attr_val|" incQuotDepth(mlir_attr_val) "]" : term
syntax "[mlir_attr_val_symbol|" mlir_attr_val_symbol "]" : term

syntax "[mlir_attr|" mlir_attr_val "]" : term -- TODO: the fuck?
macro_rules
| `([mlir_attr|  $x ]) => `([mlir_attr_val| $x ])

macro_rules
| `([mlir_attr_val| $$($x) ]) => `($x)

/-
macro_rules
| `([mlir_attr_val|  $x:numLit ]) => `(AttrVal.int $x (MLIRTy.int 64))
| `([mlir_attr_val| $x:numLit : $t:mlir_type]) => `(AttrVal.int $x [mlir_type| $t])
-/


macro_rules 
  | `([mlir_attr_val| $s:strLit]) => `(AttrVal.str $s)
  | `([mlir_attr_val| [ $xs,* ] ]) => do 
        let initList <- `([])
        let vals <- xs.getElems.foldlM (init := initList) fun xs x => `($xs ++ [[mlir_attr_val| $x]]) 
        `(AttrVal.list $vals)
 -- | `([mlir_attr_val| $ty:mlir_type]) => `(AttrVal.type [mlir_type| $ty])


/-
syntax "dense<" mlir_tensor  ">" ":" mlir_type : mlir_attr_val
macro_rules
| `([mlir_attr_val| dense< $v:mlir_tensor > : $t:mlir_type]) => 
    `(AttrVal.dense [mlir_tensor| $v] [mlir_type| $t])

macro_rules
  | `([mlir_attr_val| $a:affine_map]) =>
      `(AttrVal.affine [affine_map| $a])
-/

macro_rules
| `([mlir_attr_val_symbol| @ $x:strLit ]) =>
      `(AttrVal.symbol $x)

macro_rules
| `([mlir_attr_val_symbol| @ $x:ident ]) =>
      `(AttrVal.symbol $(Lean.quote x.getId.toString))

macro_rules
| `([mlir_attr_val_symbol| $x:mlir_attr_val_symbol :: $y:mlir_attr_val_symbol ]) =>
      `(AttrVal.nestedsymbol [mlir_attr_val_symbol| $x] [mlir_attr_val_symbol| $y])


macro_rules
| `([mlir_attr_val| $x:mlir_attr_val_symbol ]) => `([mlir_attr_val_symbol| $x])



syntax ident "=" mlir_attr_val : mlir_attr_entry
syntax strLit "=" mlir_attr_val : mlir_attr_entry

syntax "[mlir_attr_entry|" mlir_attr_entry "]" : term

-- | TODO: don't actually write an elaborator for the `ident` case. This forces
-- us to declare predefined identifiers in a controlled fashion.
macro_rules 
  | `([mlir_attr_entry| $name:ident  = $v:mlir_attr_val]) => 
     `(AttrEntry.mk $(Lean.quote (name.getId.toString))  [mlir_attr_val| $v])
  | `([mlir_attr_entry| $name:strLit  = $v:mlir_attr_val]) => 
     `(AttrEntry.mk $name [mlir_attr_val| $v])

def attrVal0Str : AttrVal := [mlir_attr_val| "add"]
#reduce attrVal0Str

def attrVal1Escape : AttrVal := [mlir_attr_val| $(attrVal0Str)]
#reduce attrVal1Escape

def AttrEntry0Escape : AttrEntry := [mlir_attr_entry| sym_name = $(attr0Str)]
#reduce AttrEntry0Escape
