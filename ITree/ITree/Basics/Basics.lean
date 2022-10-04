/-
`ITree.Basics.Basics`: General-purpose definitions
-/

abbrev arrow1 (E: Type → Type _) (F: Type → Type _) :=
  forall {T: Type}, E T → F T

infixr:40 " ~> " => arrow1

class MonadIter (M: Type → Type _) [Monad M] where
  iter: (I → M (I ⊕ R)) → I → M R

instance [Monad M] [MonadIter M]: MonadIter (StateT S M) where
  iter step i := fun s =>
    MonadIter.iter (fun (s, i) => do
       let (x, s') ← StateT.run (step i) s
       match x with
       | .inl i => return .inl (s', i)
       | .inr r => return .inr (r, s')
    ) (s, i)

instance [Monad M] [MonadIter M]: MonadIter (ReaderT S M) where
  iter step i := fun ρ =>
    MonadIter.iter (fun i => step i ρ) i

instance [Monad M] [MonadIter M]: MonadIter (OptionT M) where
  iter step i := OptionT.mk $
    MonadIter.iter (fun i => do
      let ox ← OptionT.run (step i)
      return match ox with
      | .none => .inr .none
      | .some (.inl i) => .inl i
      | .some (.inr r) => .inr (.some r)
    ) i

instance [Monad M] [MonadIter M]: MonadIter (ExceptT ε M) where
  iter step i := ExceptT.mk $
    MonadIter.iter (fun i => do
      let ex ← ExceptT.run (step i)
      return match ex with
      | .error e => .inr (.error e)
      | .ok (.inl i) => .inl i
      | .ok (.inr r) => .inr (.ok r)
    ) i

@[simp]
def equiv_pred (P Q: α → Prop) :=
  forall a, P a ↔ Q a

@[simp]
def sum_pred (P: α → Prop) (Q: β → Prop): α ⊕ β → Prop
  | .inl a => P a
  | .inr b => Q b

@[simp]
def prod_pred (P: α → Prop) (Q: β → Prop): α × β → Prop :=
  fun (a, b) => P a /\ Q b

instance: Equivalence (@equiv_pred α) where
  refl _ _ := ⟨id,id⟩
  symm H a := ⟨(H a).2, (H a).1⟩
  trans H₁ H₂ a := ⟨(H₂ a).1 ∘ (H₁ a).1, (H₁ a).2 ∘ (H₂ a).2⟩
