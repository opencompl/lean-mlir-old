declare_syntax_cat peg_toplevel
declare_syntax_cat peg_production
declare_syntax_cat peg_expr
declare_syntax_cat peg_lhs
/- 
syntax ident : peg_lhs
syntax ident: peg_expr
syntax peg_expr "+" peg_expr : peg_expr
syntax peg_expr  peg_expr : peg_expr
syntax peg_expr  "/" peg_expr : peg_expr
syntax peg_expr  "*"  : peg_expr
syntax peg_expr  "+"  : peg_expr
syntax peg_expr  "?"  : peg_expr
syntax "&" peg_expr   : peg_expr
syntax "!" peg_expr   : peg_expr
syntax "(" peg_expr ")"   : peg_expr
syntax peg_lhs "=>" peg_expr : peg_production
syntax "{" (peg_production ";")* "}": peg_toplevel
syntax "[peg|" peg_toplevel "]" : command
syntax "[peg_production|" peg_production "]" : command

macro_rules
| `([peg_production| $x:ident => $rhs:peg_expr]) =>
  `(syntax $)

macro_rules
| `([peg| $x:peg_toplevel]) => `(def foo := 1)

[peg| {
  E => E + E;
}]

 -/