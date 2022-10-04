/-
Greatest fixpoint of F, a functor which we iterate coinductively to build a
type.

The parameter `X → F X` is the destructor. From a coinductive object X, it
exposes the top-level constructor of type F X (where the argument of F is the
type of sub-objects; since it is X again, the coinduction knot is tied).

That parameter is also proof of F-consistency; since the greatest fixpoint is
the union of all F-consistent sets (sets X such that `X ⊆ F(X)`), it serves as
a witness that X is a subset of the GFP, and therefore objects of type X are
valid elements of the coinductive type.
-/
inductive ν (F: Type _ → Type _) :=
  | mk {X}: X → (X → F X) → ν F

-- Access the components
def ν.X {F}: ν F → Type _
  | @ν.mk _ X _ _ => X
def ν.x {F}: (nu: ν F) → nu.X
  | ν.mk x _ => x
def ν.f {F}: (nu: ν F) → (nu.X → F nu.X)
  | ν.mk _ f => f


/-
## Streams

Here we illustrate the generic mechanism for building constructors and
destructors. Destructors are quite easy since they're already provided.
Constructors are trickier because we need to change the set `X` as we go.
-/

-- Streams functor: takes as input a partial iteration `X` of the streams type,
-- and adds a layer of constructors.
inductive StreamsF (α: Type _) (X: Type _) :=
  | cons (a: α) (s: X)

instance: Functor (StreamsF α) where map f
  | .cons a x => .cons a (f x)

def Streams (α: Type _) := ν (StreamsF α)

-- Generic single-layer destructor and constructor (mostly boilerplate).
-- Note how in the generic constructor we want `.cons a x` to be in `X`, but it
-- might not be, so we extend `X` into `StreamsF α X`. Remember that the GFP is
-- the union of all F-consistent sets. Here we are changing the F-consistent
-- set that we use to argue that our stream is correctly constructed.
def Streams.destruct: Streams α → StreamsF α (Streams α)
  | ν.mk x f =>
      match f x with
      | .cons a x' => .cons a (ν.mk x' f)
def Streams.construct: StreamsF α (Streams α) → Streams α
  | .cons a (ν.mk x f) =>
      ν.mk (StreamsF.cons a x) (Functor.map f)

-- Specialized constructors and destructors
def Streams.cons (a: α) (s: Streams α): Streams α :=
  Streams.construct (.cons a s)
def Streams.uncons (s: Streams α): α × Streams α :=
  match s.destruct with
  | .cons a s => (a, s)
def Streams.take (n: Nat) (s: Streams α): List α :=
  match n with
  | 0 => []
  | m+1 => let (a, s') := s.uncons; a :: take m s'

-- In order to create a stream, we need to exhibit an F-consistent set, ie. a
-- type for our runtime object that we can lazily expand by `f: X → F X`.
-- `StreamsF` is super liberal so many sets fit the bill.
--
-- The function `f` does the generation, since it needs to unwrap `x: X` into
-- an `F X` which contains other occurrences of `X` - these represent the tail
-- of the stream.

-- Here we use `X = α`.
def Streams.repeat (a: α): Streams α :=
  ν.mk a (fun a => .cons a a)

-- Here we use `X = Nat`.
def Streams.integers: Streams Nat :=
  ν.mk 0 (fun n => .cons n (n+1))

-- Here we keep the original `X` and apply `f` at destruction time
def StreamsF.map (f: α → β): StreamsF α X → StreamsF β X
  | .cons a x => .cons (f a) x
def Streams.map (f: α → β): Streams α → Streams β
  | ν.mk x f0 => ν.mk x (StreamsF.map f ∘ f0)

open Streams in
section
#eval take 10 («repeat» 2)
#eval take 10 integers
#eval take 10 (map (fun x => x * x) integers)
end


/-
## ITree
-/

inductive ITreeF (E: Type → Type) (R: Type) (X: Type _) :=
  | RetF (r: R)
  | TauF (t: X)
  | VisF {T: Type} (e: E T) (k: T -> X)

def ITreeF.fmap (f: X → Y): ITreeF E R X → ITreeF E R Y
  | .RetF r => .RetF r
  | .TauF t => .TauF (f t)
  | .VisF e k => .VisF e (f ∘ k)

instance: Functor (ITreeF E R) where map := ITreeF.fmap

-- Because `VisF` quantifies on `T: Type`, we must build everything in Type 1.
-- It is quite important to fix it here by specifying `ITreeF.{1}`, otherwise
-- the universe game becomes quite complicated for no good reason.
def ITree (E: Type → Type) (R: Type) := ν (ITreeF.{1} E R)

def ITree.destruct: ITree E R → ITreeF E R (ITree E R)
  | ν.mk x f =>
      match f x with
      | .RetF r => .RetF r
      | .TauF t => .TauF (ν.mk t f)
      | .VisF e k => .VisF e (fun r => ν.mk (k r) f)

def ITree.construct: ITreeF E R (ITree E R) → ITree E R
  | .RetF r =>
      ν.mk (ULift.up r) (ITreeF.RetF ∘ ULift.down)
  | .TauF (@ν.mk _ X t f) =>
      @ν.mk _ (ITreeF E R X) (ITreeF.TauF t) (ITreeF.fmap f)
  | @ITreeF.VisF _ _ _ T e k =>
      @ν.mk _ (Option $ (t: T) × ν.X (k t)) .none
        (fun ot => match ot with
                   | .none => .VisF e (fun t => .some ⟨t, (k t).x⟩)
                   | .some ⟨t, x⟩ => Functor.map (.some ⟨t,.⟩) ((k t).f x))

def ITree.Ret (r: R): ITree E R :=
  ITree.construct (.RetF r)

def ITree.Tau (t: ITree E R): ITree E R :=
  ITree.construct (.TauF t)

def ITree.Vis (e: E T) (k: T → ITree E R): ITree E R :=
  ITree.construct (.VisF e k)

def ITreeF.cont (k: R -> ITree E U) (r: R): ITreeF E U ((r: R) × ν.X (k r)) :=
  match H: k r with
  | @ν.mk _ X x f =>
      match f x with
      | .RetF r => .RetF r
      | .TauF x => .TauF ⟨r, cast (by simp [H, ν.X]) x⟩
      | .VisF e k' => .VisF e (fun x => ⟨r, cast (by simp [H, ν.X]) (k' x)⟩)

-- Substitute every leaf `Ret x` with `k x`.
def ITree.subst (k: R -> ITree E U): ITree E R → ITree E U
  | @ν.mk _ X x f =>
      @ν.mk _ (X ⊕ ((r : R) × ν.X (k r))) (.inl x) (fun x =>
        match x with
        | .inl x =>
            match f x with
            | .RetF r => Functor.map (.inr ⟨r,.⟩) ((k r).f (k r).x)
            | .TauF t => .TauF (.inl t)
            | .VisF e k => .VisF e (.inl ∘ k)
        | .inr ⟨r, x⟩ => Functor.map (.inr ⟨r,.⟩) ((k r).f x))
