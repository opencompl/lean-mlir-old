import MLIR.Semantics.Fitree
namespace Nu
inductive Nu (F : Type _ -> Type _) : Type _ where
| mk {A} : A -> (A -> F A) -> Nu F


def Nu.get {F: Type _ -> Type _} {O: Type} (proj: {A: Type} -> F A -> O): Nu F -> O
| Nu.mk a f => proj (f a)
  

end Nu

namespace Stream
abbrev Prod S A := S × A
abbrev Stream S := Nu.Nu (Prod S)

def Stream.const {S: Type} (s: S): Stream S := 
  .mk () (fun _unit => (s, ()))

end Stream

namespace Coitree

inductive CoitreeF (EffT: Type → Type) (RetT: Type) (FixT: Type) where
  | Ret (r: RetT): CoitreeF EffT RetT FixT
  | Vis {T: Type} (e: EffT T) (k: T → FixT): CoitreeF EffT RetT FixT

-- Infinite trees?
abbrev Coitree EffT RetT := Nu.Nu (CoitreeF EffT RetT)

inductive WriteOp: Type -> Type where
| mk: String -> WriteOp Unit

open Nu in
def writeOnesForever : Coitree WriteOp Int :=
   Nu.mk () (fun unit => CoitreeF.Vis (WriteOp.mk "xx") (fun handler => ()))

/-
TODO: morphism from Fitree into Coitree
-/



/-
TODO: monad instance for Coitree 
-/


/-
TODO: morphism from Citree into Fitree, if we can show
that Coitree is well founded.
-/
end Coitree
