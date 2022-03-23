declare_syntax_cat stx1

-- | This is disturbing. Using a macro permanantly "reserves" the macro EVERYWHERE!

/-
syntax "foo" : stx1
syntax "[stx1|" stx1 "]" : term

macro_rules
| `([stx1| $x: stx1 ]) => `(1)


def x := [stx1| foo]
#reduce x
-/

def foo := 3
#reduce foo
