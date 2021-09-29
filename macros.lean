
inductive Dyck: Type :=
   | Round : Dyck -> Dyck
   | End : Dyck

declare_syntax_cat brack
syntax "End" : brack
syntax "(" brack ")" : brack
syntax "{" brack "}" : brack

-- auxiliary notation for translating `brack` into `term`
syntax "fromBrack% " brack : term

macro_rules
  | `(fromBrack% End) => `(Dyck.End)
  | `(fromBrack% ( $b )) => `(fromBrack% $b)
  | `(fromBrack% { $b }) => `(Dyck.Round (fromBrack% $b))

-- Remark: after this command `brack` will be a "reserved" keyword, and we will have to use `«brack»`
-- to reference the `brack` syntax category
macro "brack" n:ident "->" e:brack  : command =>
   `(def $n:ident : Dyck := fromBrack% $e)

brack bar -> ( End )
#print bar
/-
def bar : Dyck :=
Dyck.End
-/

brack foo -> ( { { End } } )
#print foo
