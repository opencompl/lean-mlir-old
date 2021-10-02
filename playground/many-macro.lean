-- learn how to parse multiple args in LEAN macro

inductive Dyck: Type :=
   | Round : Dyck -> Dyck
   | Flower : List (Int × Int) -> Dyck
   | Args : List Int -> Dyck
   | End : Dyck

declare_syntax_cat num_kv
syntax str "^^^" str : num_kv

syntax "num_kv% " num_kv : term
macro_rules
  | `(num_kv% $i ^^^  $j) => `( ($i , $j) )

def foo := (num_kv% "foo"  ^^^  "bar")
#print foo

declare_syntax_cat brack
syntax "End" : brack
syntax "(" brack ")" : brack
syntax "{" num_kv "}" : brack
-- syntax "{" sepBy(term, ",") "}" : brack
syntax "<[" "]>" : brack
syntax "<[" sepBy(term, ",") "]>" : brack

syntax "<" term "," term,* ">" : brack
-- auxiliary notation for translating `brack` into `term`

syntax "fromBrack% " brack : term

-- set_option trace.Elab.definition true in
-- | init/data/array/basic.lean for sepBy
macro_rules
  | `(fromBrack% End) => `(Dyck.End)
  | `(fromBrack% ( $b )) => `(Dyck.Round (fromBrack% $b))
  | `(fromBrack% { $i:num_kv }) => `(Dyck.Flower (num_kv% $i))
  | `(fromBrack% <[ ]>) => `(Dyck.Args [])
  | `(fromBrack% <[ $js,* ]> ) => `(Dyck.Args [ $js,* ])

-- Remark: after this command `brack` will be a "reserved" keyword, and we will have to use `«brack»`
-- to reference the `brack` syntax category
macro "brack" n:ident "->" e:brack  : command =>
   `(def $n:ident : Dyck := (fromBrack% $e))

brack foo ->  { 1::2  }
#print foo 

brack baz -> <[ ]>
#print baz

brack baz2 -> <[ 1, 2, 3]>
#print baz2

