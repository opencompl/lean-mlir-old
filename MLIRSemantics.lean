/- Extendable effect families -/

def to1 (E F: Type → Type) :=
  ∀ T, E T → F T
def sum1 (E F: Type → Type) :=
  fun T => E T ⊕ F T
inductive Void1: Type -> Type

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

inductive StateE {S: Type}: Type → Type where
  | Read: Unit → StateE S
  | Write: S → StateE Unit

inductive WriteE {W: Type}: Type → Type where
  | Tell: W → WriteE Unit


/- The monadic domain; essentially finite Interaction Trees -/

inductive Fitree (E: Type → Type) (R: Type) where
  | Ret (r: R): Fitree E R
  | Vis {T: Type} (e: E T) (k: T → Fitree E R): Fitree E R

def Fitree.ret {E R}: R → Fitree E R :=
  Fitree.Ret

def Fitree.trigger {E F: Type → Type} {T} [Member E F] (e: E T): Fitree F T :=
  Fitree.Vis (Member.inject _ e) Fitree.ret

def Fitree.bind {E R T} (t: Fitree E T) (k: T → Fitree E R) :=
  match t with
  | Ret r => k r
  | Vis e k' => Vis e (λ r => bind (k' r) k)

instance {E}: Monad (Fitree E) where
  pure := Fitree.ret
  bind := Fitree.bind


-- Interpretation into the monad of finite ITrees
def interp {M} [Monad M] {E} (h: forall ⦃T⦄, E T → M T):
    forall ⦃R⦄, Fitree E R → M R :=
  λ _ t =>
    match t with
    | Fitree.Ret r => pure r
    | Fitree.Vis e k => bind (h e) (λ t => interp h (k t))

-- Interpretation into the state monad
def interp_state {M S} [Monad M] {E} (h: forall ⦃T⦄, E T → StateT S M T):
    forall ⦃R⦄, Fitree E R → StateT S M R :=
  interp h

-- Since we only use finite ITrees, we can actually run them when they're
-- fully interpreted (which leaves only the Ret constructor)
def Fitree.run {R}: Fitree Void1 R → R
  | Ret r => r
  | Vis e k => nomatch e
