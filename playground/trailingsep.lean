import Lean
import Lean.Parser
open Lean
open Lean.Parser

-- we want to parse terms of the form ["foo" × "bar" × ... ×  "quux" ×  10]
declare_syntax_cat bar
syntax sepBy(str, "×") "×" num : bar
syntax "[bar|" bar "]" : term

-- | ERROR: expected × 
macro_rules
| `([bar| $[ $xs ]×* × $y ]) => return quote xs

-- | ERROR: expected × 
macro_rules
| `([bar| $[ $xs ]×* $y ]) => return quote xs

-- vvv what does work vvv
-- The macro without the trailing ×  num works!
declare_syntax_cat foo
syntax sepBy(str, "×")  : foo
syntax "[foo|" foo "]" : term

macro_rules
| `([foo| $[ $xs ]×* ]) => return quote xs

