inductive Dyck: Type := 
   | Round : Dyck -> Dyck 
   | End : Dyck

declare_syntax_cat brack
syntax "End" : brack
syntax "(" brack ")" : brack
syntax "{" brack "}" : brack

macro "End" : term => `(Dyck.End)
macro "(" expr:brack ")" : term => `(Dyck.Round $expr)

syntax "brack" ident "->" brack : command

macro "brack" n:ident "End": command => 
   `(def $n : Dyck := End)

-- | macro that wants `( brack )`
macro "brack" n:ident "->" "(" e:brack ")" : command => 
   `(def $n : Dyck := Dyck.Round $e)

brack foo End
print foo

brack bar ( End )
print bar

