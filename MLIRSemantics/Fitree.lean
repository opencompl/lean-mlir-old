/- Extendable effect families -/

section events
universe u v

def pto (E: Type → Type u) (F: Type → Type v) :=
  ∀ T, E T → F T
def psum (E: Type → Type u) (F: Type → Type v) :=
  fun T => E T ⊕ F T
inductive PVoid: Type -> Type u

infixr:40 " ~> " => pto

class Member (E: Type → Type u) (F: Type → Type v) where
  inject : E ~> F

instance {E}: Member E E where
  inject := (fun _ => id)

instance {E F G} [Member E F]: Member E (psum F G) where
  inject T := Sum.inl ∘ Member.inject T

instance {E F G} [Member E G]: Member E (psum F G) where
  inject T := Sum.inr ∘ Member.inject T

-- Effects can now be put in context automatically by typeclass resolution
example (E: Type → Type u):
  Member E E := inferInstance
example (E: Type → Type u) (F: Type → Type v):
  Member E (psum E F) := inferInstance
example (E: Type → Type u) (F: Type → Type v):
  Member E (psum F (psum F E)) := inferInstance

end events


/- Examples of interactions -/

inductive StateE {S: Type}: Type → Type where
  | Read: Unit → StateE S
  | Write: S → StateE Unit

inductive WriteE {W: Type}: Type → Type where
  | Tell: W → WriteE Unit


/- The monadic domain; essentially finite Interaction Trees -/

section fitree
universe u v

inductive Fitree (E: Type → Type u) (R: Type) where
  | Ret (r: R): Fitree E R
  | Vis {T: Type} (e: E T) (k: T → Fitree E R): Fitree E R

def Fitree.ret {E R}: R → Fitree E R :=
  Fitree.Ret

def Fitree.trigger {E: Type → Type u} {F: Type → Type v} {T} [Member E F]
    (e: E T): Fitree F T :=
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
def Fitree.run {R}: Fitree PVoid R → R
  | Ret r => r
  | Vis e k => nomatch e

end fitree
