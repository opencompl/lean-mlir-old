-- example on parsing arith language via macros
inductive Arith: Type :=
   | Add : Arith -> Arith -> Arith
   | Int : Int -> Arith
   | Symbol : String -> Arith
   | Mul : Arith -> Arith -> Arith

declare_syntax_cat arith
syntax term : arith
syntax  "END" : arith
syntax  arith ":+" arith : arith
syntax arith ":*" arith : arith
syntax "(" arith ")" : arith

-- auxiliary notation for translating `arith` into `term`
syntax "fromArith% " arith : term

macro_rules
  | `(fromArith% $num:term) => `(Arith.Symbol $num)
  | `(fromArith% END) => `(Arith.Int 50)
  | `(fromArith% $x:arith :+ $y:arith ) => `(Arith.Add (fromArith% $x) (fromArith% $y))
  | `(fromArith% $x:arith :* $y:arith ) => `(Arith.Mul (fromArith% $x) (fromArith% $y))
  | `(fromArith% ($x:arith)) => `(fromArith% $x)

-- Remark: after this command `brack` will be a "reserved" keyword, and we will have to use `«brack»`
-- to reference the `brack` syntax category
macro "arith" n:ident "->" e:arith  : command =>
   `(def $n:ident : Arith := fromArith% $e)

arith bar -> END
#print bar

arith foo -> "x" :* "y"
#print foo

arith baz -> "x" :+ "y"
#print baz

arith baz2 -> ("x" :+ "y")
#print baz2

arith baz3 -> ("x" :+ ("z" :* "y"))
#print baz3


syntax ident : «arith»  -- Have to use french quotes since `arith` is now a keyword

macro_rules
  | `(fromArith% $x:ident) => `(Arith.Symbol $(Lean.quote (toString x.getId)))

arith foo2 -> x 
#print foo2

arith foo3 -> x :+ y
#print foo3

-- arith foo2 -> x + y + z
-- #print foo2
 
syntax "{" term "}" : «arith» -- escape for embedding terms into `Arith`
 
macro_rules
  | `(fromArith% { $e }) => e

arith boo -> {foo3}
#print boo
-- 
