import MLIR.Util.Tactics

def ExceptMonad.split {x: Except ε α} {y: α -> Except ε α}:
  (do let t <- x; y t) = .ok v -> ∃ x_v, x = .ok x_v := by
  cases x <;> intro H <;> try contradiction
  rename_i a
  exists a

def ExceptMonad.simp_ok {y: α -> Except ε α}:
  (do let t <- .ok x; y t) = y x := by rfl

def List.mapM_nil {m : Type u → Type v} [Monad m] {α : Type w} {β : Type u} (f : α → m β):
    List.mapM f [] = pure [] := by rfl

-- Is it possible to prove this? Do we need lawful monad for this?
def List.mapM_cons {m : Type u → Type v} [Monad m] {α : Type w} {β : Type u} (f : α → m β) (head: α) (tail: List α):
    List.mapM f (head::tail) = (do
         let head' ← f head
         let tail' ← mapM f tail
         return head'::tail') := by
  sorry

/- Simp and unfold all ExceptMonad and monad definitions -/

macro "simp_monad" : tactic =>   
  `(tactic| repeat (progress 
     (try (rw [List.mapM_nil]);
      try (rw [List.mapM_cons]);
      simp [pure, StateT.pure, Except.pure, bind, StateT.bind, Except.bind,
            get, StateT.get, set, StateT.set])))

macro "simp_monad" "at" Hname:ident : tactic =>   
  `(tactic| repeat (progress 
     (try (rw [List.mapM_nil] at $Hname:ident);
      try (rw [List.mapM_cons] at $Hname:ident);
      simp [pure, StateT.pure, Except.pure, bind, StateT.bind, Except.bind,
            get, StateT.get, set, StateT.set] at $Hname:ident)))

macro "simp_monad" "at" "*" : tactic =>
  `(tactic| repeat (progress 
     (try (rw [List.mapM_nil] at *);
      try (rw [List.mapM_cons] at *);
      simp [pure, StateT.pure, Except.pure, bind, StateT.bind, Except.bind,
            get, StateT.get, set, StateT.set, getThe, MonadStateOf.get] at *)))
