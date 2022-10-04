/-
## ITree
-/

import ITree.Basics.Basics

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

def ν.X {F}: ν F → Type _
  | @ν.mk _ X _ _ => X
def ν.x {F}: (nu: ν F) → nu.X
  | ν.mk x _ => x
def ν.f {F}: (nu: ν F) → (nu.X → F nu.X)
  | ν.mk _ f => f

-- The functor `ITreeF` takes as parameter partially-iterated type `X` and adds
-- one of infinitely many constructor applications.
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

abbrev ITree' E R := ITreeF E R (ITree E R)

def ITree.observe: ITree E R → ITree' E R
  | ν.mk x f =>
      match f x with
      | .RetF r => .RetF r
      | .TauF t => .TauF (ν.mk t f)
      | .VisF e k => .VisF e (fun r => ν.mk (k r) f)

-- We materialize Ret nodes using the return type as base. Because everything
-- lives in `Type 1` we lift to `ULift R`.
def ITree.Ret (r: R): ITree E R :=
  ν.mk (ULift.up r) (ITreeF.RetF ∘ ULift.down)

-- We materialize Tau nodes as an `ITreeF E R X`. Basically, we pre-expand one
-- level of the object; and we initialize that process by adding a `.TauF` as
-- the first pre-expanded level.
def ITree.Tau: ITree E R → ITree E R
  | @ν.mk _ X x f =>
      @ν.mk _ (ITreeF E R X) (.TauF x) (Functor.map f)

-- We materialize Vis nodes as dependent pairs, which allows us to remember
-- which `t: T` was used to enter `k`, and thus access the associated
-- destructor `(k t).f`. We also wrap it in an option to that the first
-- destruction can produce the `Vis` node, and later transition into `k`.
def ITree.Vis (e: E T) (k: T → ITree E R): ITree E R :=
  @ν.mk _ (Option $ (t: T) × ν.X (k t)) .none fun
    | .none => .VisF e (fun t => .some ⟨t, (k t).x⟩)
    | .some ⟨t, x⟩ => Functor.map (.some ⟨t,.⟩) ((k t).f x)

def ITree'.make: ITree' E R → ITree E R
  | .RetF r => .Ret r
  | .TauF t => .Tau t
  | .VisF e k => .Vis e k

-- Substitute every leaf `Ret x` with `k x`. Similar to Vis, we keep a
-- dependent pair to track the continuation's branch. Before we reach the
-- continuation we stay on the supporting type `X` of the original tree.
def ITree.subst (k: R -> ITree E U): ITree E R → ITree E U
  | @ν.mk _ X x f =>
      @ν.mk _ (X ⊕ ((r : R) × ν.X (k r))) (.inl x) fun
        | .inl x =>
            match f x with
            -- When transitioning into `k` we immediately unfold the first
            -- level since we don't want to return the `RetF r`.
            | .RetF r => Functor.map (.inr ⟨r,.⟩) ((k r).f (k r).x)
            | .TauF t => .TauF (.inl t)
            | .VisF e k => .VisF e (.inl ∘ k)
        | .inr ⟨r, x⟩ => Functor.map (.inr ⟨r,.⟩) ((k r).f x)

def ITree.bind (t: ITree E T) (k: T → ITree E R): ITree E R :=
  subst k t

def ITree.cat (k: T → ITree E U) (h: U → ITree E V): T → ITree E V :=
  fun t => bind (k t) h

-- Iterate `step i` for as long as it returns `inl i`. Once it returns `inr r`,
-- return `r`. In addition to the current object from `ν.X (step i)`, we store
-- the current `i` to be able to access the destructor.
def ITree.iter (step: I → ITree E (I ⊕ R)) (i: I): ITree E R :=
  @ν.mk _ ((i: I) × ν.X (step i)) ⟨i, ν.x (step i)⟩ fun ⟨i, x⟩ =>
    match (step i).f x with
    -- When transitioning we introduce a `Tau` to avoid an infinite loop if
    -- `step i'` where to immediately into `RetF`, infinitely.
    | .RetF (.inl i') => .TauF ⟨i', (step i').x⟩
    | .RetF (.inr r) => .RetF r
    | .TauF x => .TauF ⟨i, x⟩
    | .VisF e k => .VisF e (fun x => ⟨i, k x⟩)

def ITree.map (f: R → S) (t: ITree E R): ITree E S :=
  ITree.bind t (Ret ∘ f)

def ITree.trigger: E ~> ITree E :=
  fun e => Vis e Ret

def ITree.ignore: ITree E R → ITree E Unit :=
  ITree.map (fun _ => ())

def ITree.spin: ITree E R :=
  ν.mk PUnit.unit (fun _ => .TauF .unit)

def ITree.forever (t: ITree E R): ITree E R :=
  ν.mk t.x (fun x =>
    match t.f x with
    | .RetF _ => .TauF t.x
    | .TauF x' => .TauF x'
    | .VisF e k => .VisF e k)

instance: Functor (ITree E) where
  map := ITree.map

instance: Monad (ITree E) where
  pure := ITree.Ret
  bind := ITree.bind

instance: MonadIter (ITree E) where
  iter := ITree.iter

def ITree.burn (n: Nat) (t: ITree E R): ITree E R :=
  match n with
  | 0 => t
  | m+1 =>
      match observe t with
      | .RetF r => Ret r
      | .VisF e k => Vis e k
      | .TauF t' => burn m t'
