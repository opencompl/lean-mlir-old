/- Extendable effect families -/

def to1 (E F: Type → Type) :=
  ∀ T, E T → F T
def sum1 (E F: Type → Type) :=
  fun T => E T ⊕ F T
inductive void1: Type -> Type

infixr:40 " ~> " => to1

class Member (E F: Type → Type) where
  inject : E ~> F

instance {E}: Member E E where
  inject := (fun _ => id)

instance {E F G} [Member E F]: Member E (sum1 F G) where
  inject T := Sum.inl ∘ Member.inject T

instance {E F G} [Member E G]: Member E (sum1 F G) where
  inject T := Sum.inr ∘ Member.inject T

-- Effects can now be put in context automatically by typeclass resolution
example (E: Type → Type): Member E E := inferInstance
example (E F: Type → Type): Member E (sum1 E F) := inferInstance
example (E F: Type → Type): Member E (sum1 F (sum1 F E)) := inferInstance


/- Examples of interactions -/

inductive stateE {S: Type}: Type → Type where
  | Read: Unit → stateE S
  | Write: S → stateE Unit

inductive writeE {W: Type}: Type → Type where
  | Tell: W → writeE Unit


/- The monadic domain; essentially finite Interaction Trees -/

inductive fitree (E: Type → Type) (R: Type) where
  | Ret (r: R): fitree E R
  | Vis {T: Type} (e: E T) (k: T → fitree E R): fitree E R

def fitree.ret {E R}: R → fitree E R :=
  fitree.Ret

def fitree.trigger {E F: Type → Type} {T} [Member E F] (e: E T): fitree F T :=
  fitree.Vis (Member.inject _ e) fitree.ret

def fitree.bind {E R T} (t: fitree E T) (k: T → fitree E R) :=
  match t with
  | Ret r => k r
  | Vis e k' => Vis e (λ r => bind (k' r) k)

instance {E}: Monad (fitree E) where
  pure := fitree.ret
  bind := fitree.bind


-- Interpretation into the monad of finite ITrees
def interp {M} [Monad M] {E} (h: forall ⦃T⦄, E T → M T):
    forall ⦃R⦄, fitree E R → M R :=
  λ _ t =>
    match t with
    | fitree.Ret r => pure r
    | fitree.Vis e k => bind (h e) (λ t => interp h (k t))

-- Interpretation into the state monad
def interp_state {M S} [Monad M] {E} (h: forall ⦃T⦄, E T → StateT S M T):
    forall ⦃R⦄, fitree E R → StateT S M R :=
  interp h

-- Since we only use finite ITrees, we can actually run them when they're
-- fully interpreted (which leaves only the Ret constructor)
def fitree.run {R}: fitree void1 R → R
  | Ret r => r
  | Vis e k => nomatch e
