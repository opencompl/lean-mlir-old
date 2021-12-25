-- https://mtzguido.github.io/pubs/dm4all.pdf

-- | a specification is a monad morphism from a computation monad into a specification monad.
class MonadMorphism (M: Type -> Type) (W: Type -> Type) {Mmonad: Monad M} {Wmonad : Monad W} where
  θ : ∀ {T: Type}, M T -> W T
  -- | theta transports pure
  theta_pure : forall {T: Type} (a: T), θ (pure a) = pure a
  -- | theta distributes over bind
  theta_bind : forall {S T: Type} (a: M S) (f: S -> M T), θ (a >>= f) = θ a >>= (θ ∘ f)


inductive Id' (A: Type): Type where
| mk: (a: A) -> Id' A

instance : Monad Id' where 
  pure {A: Type} (a: A) := Id'.mk a
  bind {A B : Type} (ma: Id' A) (a2mb: A -> Id' B) : Id' B :=
   match ma with
   | Id'.mk a => a2mb a


inductive PureSpec (A: Type) : Type where
| mk: (post2pre: (A -> Prop) -> Prop) -> PureSpec A


def PureSpec.run {A: Type} (post2pre: PureSpec A) (post: A -> Prop): Prop :=
  match post2pre with
  | PureSpec.mk post2pre => post2pre post
  
instance : Monad PureSpec where
  pure {A: Type} (a: A) := PureSpec.mk (fun post => post a)
  bind {A B: Type} (post2prea: PureSpec A) (a2post2preb: A -> PureSpec B): PureSpec B :=
     PureSpec.mk (fun postb => post2prea.run (fun a => (a2post2preb a).run postb))

-- | how to disable prelude?
inductive MaybeT (M: Type) (A: Type): Type where
 | Just: (a: A) -> MaybeT M A
 | Nothing: MaybeT M A

