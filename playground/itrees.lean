inductive itreeF (E: Type -> Type) (R: Type) (itree : Type) :=
| RetF: (r : R) -> itreeF E R itree
| TauF: (t : itree) -> itreeF E R itree
| VisF: (X : Type) -> (e : E X)  -> (k : X -> itree) -> itreeF E R itree

/--/
This presents the traces of effectful programs as trees where
each node issues a command c from signature C and receives a
suitable response in R c. For more on this way of looking at
interaction, see papers by Peter Hancock, Anton Setzer, and
others.
-/

Inductive FreeMonad (C:Type) (R : C -> Type) (X:Type): Type:=
| ret: X -> FreeMonad C R X
| eff: forall c:C, R c -> FreeMonad C R X.


/-
A more expensive, but perhaps more rewarding approach is to
define a *universe* of strictly positive functors, interpreted
via an inductive family. You can read more about this approach
in papers by Peter Morris and colleagues.
-/