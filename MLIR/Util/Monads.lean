import MLIR.Util.Tactics
import Std.Data.List.Basic
import Std.Data.List.Lemmas

def ExceptMonad.split {x: Except ε α} {y: α -> Except ε α}:
  (do let t <- x; y t) = .ok v -> ∃ x_v, x = .ok x_v := by
  cases x <;> intro H <;> try contradiction
  rename_i a
  exists a

def ExceptMonad.simp_ok {y: α -> Except ε α}:
  (do let t <- .ok x; y t) = y x := by rfl

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
