import Lean.Parser
open Lean

inductive Arith: Type :=
   | Add : Arith -> Arith -> Arith
   | Symbol : String -> Arith

declare_syntax_cat arith
syntax arith "+" arith : arith

syntax:max "fromArith% " arith : term

macro_rules
  | `(fromArith% $x:arith + $y:arith ) => `(Arith.Add (fromArith% $x) (fromArith% $y))

macro "arith" n:ident "->" e:arith  : command =>
   `(def $n:ident : Arith := fromArith% $e)


syntax ident : «arith»  -- Have to use french quotes since `arith` is now a keyword

macro_rules
  | `(fromArith% $x:ident) => `(Arith.Symbol $(Lean.quote (toString x.getId)))

arith foo -> x + y + z
#print foo


syntax "{" term "}" : «arith» -- escape for embedding terms into `Arith`

macro_rules
  | `(fromArith% { $e }) => e

arith boo -> x + y + {foo}
#print boo

