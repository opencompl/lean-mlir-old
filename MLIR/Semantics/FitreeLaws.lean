import MLIR.Semantics.Fitree

theorem Fitree.map_const:
    (Functor.mapConst: R₁ → Fitree E R₂ → Fitree E R₁)
    = Functor.map ∘ Function.const R₂ :=
  rfl

theorem Fitree.id_map (t: Fitree E R):
    id <$> t = t := by
  simp [Functor.map]
  induction t with
  | Ret _ => rfl
  | Vis _ _ ih => simp [bind, ih]

theorem Fitree.comp_map (f: R₁ → R₂) (g: R₂ → R₃) (t: Fitree E R₁):
    (g ∘ f) <$> t = g <$> f <$> t := by
  simp [Functor.map]
  induction t with
  | Ret _ => rfl
  | Vis _ _ ih => simp [bind, ih]

instance {E}: LawfulFunctor (Fitree E) where
  map_const  := Fitree.map_const
  id_map     := Fitree.id_map
  comp_map   := Fitree.comp_map

theorem Fitree.seqLeft_eq (t₁: Fitree E R₁) (t₂: Fitree E R₂):
    t₁ <* t₂ = Function.const R₂ <$> t₁ <*> t₂ := by
  simp [SeqLeft.seqLeft, Seq.seq]
  induction t₁ with
  | Ret _ => rfl
  | Vis _ _ ih => simp [bind]; funext _; simp [ih]; rfl

theorem Fitree.seqRight_eq (t₁: Fitree E R₁) (t₂: Fitree E R₂):
    t₁ *> t₂ = Function.const R₁ id <$> t₁ <*> t₂ := by
  simp [SeqRight.seqRight, Seq.seq]
  induction t₁ with
  | Ret _ =>
    simp [Function.const, Function.comp, bind]
    induction t₂ with
    | Ret _ => rfl
    | Vis _ _ ih₂ => simp [bind, ←ih₂]
  | Vis _ _ ih =>
    simp [bind]; funext _; simp [ih]; rfl

theorem Fitree.pure_seq (f: R₁ → R₂) (t: Fitree E R₁):
    pure f <*> t = f <$> t :=
  rfl

theorem Fitree.map_pure (f: R₁ → R₂) (r: R₁):
    f <$> (pure r) = pure (f := Fitree E) (f r) :=
  rfl

theorem Fitree.seq_pure (f: Fitree E (R₁ → R₂)) (r: R₁):
    f <*> pure r = (fun h => h r) <$> f :=
  rfl

theorem Fitree.seq_assoc (t₁: Fitree E R₁)
    (t₂: Fitree E (R₁ → R₂)) (t₃: Fitree E (R₂ → R₃)):
    t₃ <*> (t₂ <*> t₁) = ((@Function.comp R₁ R₂ R₃) <$> t₃) <*> t₂ <*> t₁ := by
  sorry

instance {E}: LawfulApplicative (Fitree E) where
  seqLeft_eq   := Fitree.seqLeft_eq
  seqRight_eq  := Fitree.seqRight_eq
  pure_seq     := Fitree.pure_seq
  map_pure     := Fitree.map_pure
  seq_pure     := Fitree.seq_pure
  seq_assoc    := Fitree.seq_assoc


theorem Fitree.bind_pure_comp (f: R₁ → R₂) (t: Fitree E R₁):
    bind t (fun r => pure (f r)) = f <$> t :=
  rfl

theorem Fitree.bind_map (f: Fitree E (R₁ → R₂)) (t: Fitree E R₁):
    bind f (. <$> t) = f <*> t :=
  rfl

theorem Fitree.pure_bind (r: R₁) (k: R₁ → Fitree E R₂):
    bind (pure r) k = k r :=
  rfl

theorem Fitree.bind_assoc (t: Fitree E R₁)
    (k₁: R₁ → Fitree E R₂) (k₂: R₂ → Fitree E R₃):
    bind (bind t k₁) k₂ = bind t (fun x => bind (k₁ x) k₂) := by
  induction t with
  | Ret _ => rfl
  | Vis _ _ ih => simp [bind]; funext _; simp [ih]

instance {E}: LawfulMonad (Fitree E) where
  bind_pure_comp  := Fitree.bind_pure_comp
  bind_map        := Fitree.bind_map
  pure_bind       := Fitree.pure_bind
  bind_assoc      := Fitree.bind_assoc
