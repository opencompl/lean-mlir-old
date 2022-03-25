import Lean.Parser
import Lean.Parser.Extra

open Lean
open Lean.Parser


declare_syntax_cat foo
syntax ident : foo 
syntax "[foo|" ident "]" : term

macro_rules
| `([foo| $x:ident ]) => `("foo")    


declare_syntax_cat bar
syntax foo : bar 
syntax "[bar|" ident "]" : term


def fn (x: Syntax): MacroM Syntax := 
    let k := Lean.mkIdent x.getId.toString
    `([foo| $(Lean.quote k)])

macro_rules
| `([bar| $x:ident]) => 
      fn x



def fooExample := [foo| x]
#reduce fooExample

def barExample := [bar| x]
#reduce barExample