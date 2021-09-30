-- set_option trace.Elab.definition true

inductive T: Type :=
   | mkT : T

macro "pleaseMkT" : command =>
   `(def n : T := T.mkT)


def global_t : T := T.mkT

-- | names are not reuse
pleaseMkT
pleaseMkT
