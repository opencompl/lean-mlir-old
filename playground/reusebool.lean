-- see how we can use bool both in custom syntax
-- and in "regular" Lean syntax
import Lean.Parser
open Lean.Parser


declare_syntax_cat foo (leadingIdentBehavior := LeadingIdentBehavior.symbol)
def trueVal  :=  nonReservedSymbol "true"
  
syntax "[foo|" foo  "]" : term
syntax  &"true" : foo

macro_rules
| `([foo| $x:foo ]) => `("true")

def succeeds : String := [foo| true ]
def fails : Bool := true