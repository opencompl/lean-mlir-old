import Lean
import Lean.Parser
open Lean
open Lean.Parser


declare_syntax_cat static_dim_list
syntax "[static_dim_list|" static_dim_list "]" : term
-- syntax sepBy(numLit, "×") : static_dim_list
syntax sepBy(numLit, "×", "×" notFollowedBy(strLit)) : static_dim_list

macro_rules
| `([static_dim_list| $[ $ns:numLit ]×* ]) => `([ $ns,* ])

def staticDimList1 : List Int := [static_dim_list| 1 × 2]
#reduce staticDimList1

declare_syntax_cat mlir_type
syntax "vector" "<" (static_dim_list "×" ("[" static_dim_list "]" "×")? )?  str ">"  : mlir_type
syntax "[mlir_type|" mlir_type "]" : term 

inductive MLIRType 
| vec: List Int -> List Int -> String -> MLIRType

-- | We need b to know that it is Option Syntax when a has Syntax inside it!
macro_rules
| `([mlir_type| vector < $[ $a?:static_dim_list × $[ [ $b?:static_dim_list ] × ]? ]? $t:strLit  >]) => do
      let a <- match a? with  
        | some s =>  `([static_dim_list| $s])
        | none => `((@List.nil Int))
      let b <- match b? with  
        | some (some s) => `([static_dim_list| $s])
        | _ => `((@List.nil Int))
      `(MLIRType.vec $a $b $t)

-- | error: expected numeral
def vectorTy1 := [mlir_type| vector<2 × "i32">]
#print vectorTy1
/-
-- | TODO: Do I need two different parsers for these cases?
declare_syntax_cat mlir_type
syntax "memref" "<"  "*" "×" mlir_type ("," mlir_attr_val)?  ">"  : mlir_type
syntax "memref" "<*" "×" mlir_type ("," mlir_attr_val)?  ">"  : mlir_type
-/



/- structs do not allow `default` in their creation.
  let ff : CtorView := default
  let ctorError : CtorView := {default with inferMod := true} <- ERRORS OUT
  let ctor0: CtorView := { ff with inferMod := true}
-/
