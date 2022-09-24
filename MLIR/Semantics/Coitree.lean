import MLIR.Semantics.Fitree
namespace Nu

/-
A representation of an infinite process, whose output
space is parametrized by F, where the A encodes
the state space.

We think of Nu as modelling a machine, whose current 
(hidden) state is (a: A), and has a transition function
(A -> F A). This output (F A) has some way to extract an 'A',
so we can run the machine another step, along with some auxiliary
data of interest, which forms the output of the Nu. The auxiliary
output is chosen to produce the data structure on is interested in.

For example, (F A := O × A) produces a way to talk about shapes
where the output value at this step is 'O', and A is the current state.
-/
inductive Nu (F : Type _ -> Type _) : Type _ where
| mk (A) : A -> (A -> F A) -> Nu F

def Nu.get {F: Type _ -> Type _} {O: Type} (proj: {A: Type} -> F A -> O): Nu F -> O
| Nu.mk A a f => proj (f a)

def Nu.getSigma {F: Type _ -> Type _} (nu: Nu F): Σ(A: Type), (A × (A -> F A)) :=
match nu with
| Nu.mk A a f => ⟨A, (a, f)⟩

end Nu

namespace Stream
/-
We define the type of infinite streams using Nu.
-/
abbrev Prod S A := S × A
abbrev Stream S := Nu.Nu (Prod S)

def Stream.repeat {S: Type} (s: S): Stream S := 
  .mk Unit () (fun _unit => (s, ()))

end Stream

namespace  PossiblyInfiniteList
/-
We try to define the type of Possibly Infinite Lists (PILists).
The F is a haskell convention of the functor whose Nu gives the
actual value of interest. A value-level analogy is:

```hs
def factorialF (f: Int -> Int) (v: Int): Int := 
  if v == 0 then 1 else v * f (v - 1)
def factorial: Int -> Int := nu factorialF
```

(ie, a type that could have as inhabitants both finite lists and infinite streams)
-/
inductive PIListF (S: Type): Type -> Type
| nil {A}: PIListF S A
| cons: S -> A -> PIListF S A

def PIListF.map (f: A -> B): PIListF S A -> PIListF S B
| .nil => .nil
| .cons s a => .cons s (f a)

abbrev PIList S := Nu.Nu (PIListF S)

open Nu in 
def PIList.nil: PIList A :=
   Nu.mk Unit () (fun () => .nil)

/-
We encode the state of the new machine, essentially in unary,
by adding a new Option layer to the state.
When the state is '.none', we return the new element 'a'.
When the state is '.some', we run the previous machine in 'as'.
-/
open Nu in 
def PIList.cons (a: A) (as: PIList A): PIList A :=
   let ⟨S, (s, f)⟩ := as.getSigma
   Nu.mk (Option S) .none 
     (fun os => match os with
                | .none => .cons a (.some s)
                | .some s => (f s).map .some)


-- Project a single layer of the PIList out.
open Nu in 
def PIList.destruct (as: PIList A): Option (A × PIList A) :=
   let ⟨S, (s, f)⟩ := as.getSigma
   match f s with 
   | .nil => .none
   | .cons a s' => .some (a, Nu.mk S s' f)


-- An infinite repeating stream
def PIList.repeat {S: Type} (s: S): PIList S := 
  .mk Unit () (fun _unit => PIListF.cons s ())

-- Use the previously encoded combinators to convert
-- a List into a PIList.
def PIList.ofList: List S -> PIList S 
| .nil => PIList.nil
| .cons a as => PIList.cons a (PIList.ofList as)

/-
Try to convert the PIList to a list.
If there is enough fuel, it returns (.some <list>), and .none
if there isn't.
-/
def PIList.toList (n: Nat) (ss: PIList S): Option (List S) :=
match n with
| 0 => .none
| n' + 1 => 
   match ss.destruct with 
   | .none => .some .nil
   | .some (s, ss') => (PIList.toList n' ss').map (.cons s)
end PossiblyInfiniteList

namespace Coitree

inductive CoitreeF (EffT: Type → Type) (RetT: Type) (FixT: Type) where
  | Ret (r: RetT): CoitreeF EffT RetT FixT
  | Vis {T: Type} (e: EffT T) (k: T → FixT): CoitreeF EffT RetT FixT

abbrev Coitree EffT RetT := Nu.Nu (CoitreeF EffT RetT)


open Nu in 
def Coitree.Vis (e: E T) (k: T -> Coitree E R): Coitree E R :=
  Nu.mk Unit () (fun unit => CoitreeF.Vis e (fun v => k v))


open Nu in 
def Coitree.Ret (r: T): Coitree E T :=
  Nu.mk Unit () (fun unit => CoitreeF.Ret r)

/-
Test that we can build infinite Coitrees of effects.
We build an infinite sequence of writes.
-/
inductive WriteOp: Type -> Type where
| mk: String -> WriteOp Unit

open Nu in
def writeOnesForever : Coitree WriteOp Int :=
   Nu.mk Unit () (fun unit => CoitreeF.Vis (WriteOp.mk "xx") (fun handler => ()))


inductive CoitreeLayer EffT RetT
| Ret: RetT -> CoitreeLayer EffT RetT
| Vis: (e: EffT T) -> (k: T -> Coitree EffT RetT) -> CoitreeLayer EffT RetT

/-
Destruct a layer of a Coitree.
TODO: think this through carefully.
-/
open Nu in
def Coitree.destruct (as: Coitree E T):CoitreeLayer E T :=
   let ⟨S, (s, f)⟩ := as.getSigma
   match f s with 
   | .Ret r => .Ret r
   | .Vis e k => .Vis e (fun t => Nu.mk S (k t) f)



/-
Morphism from Fitree into a Coitree.
-/
def mor: Fitree E R -> Coitree E R 
| Fitree.Ret r => Coitree.Ret r
| Fitree.Vis et k => Coitree.Vis et (fun t => mor (k t))


end Coitree

namespace LawfulCoitree
open Coitree


/-
theorem Coitree.map_const:
    (Functor.mapConst: R₁ → Coitree E R₂ → Coitree E R₁)
    = Functor.map ∘ Function.const R₂ := sorry

theorem Coitree.id_map (t: Coitree E R):
    id <$> t = t := by sorry

theorem Coitree.comp_map (f: R₁ → R₂) (g: R₂ → R₃) (t: Coitree E R₁):
    (g ∘ f) <$> t = g <$> f <$> t := by sorry

instance {E}: LawfulFunctor (Coitree E) where
  map_const  := Coitree.map_const
  id_map     := Coitree.id_map
  comp_map   := Coitree.comp_map

theorem Coitree.seqLeft_eq (t₁: Coitree E R₁) (t₂: Coitree E R₂):
    t₁ <* t₂ = Function.const R₂ <$> t₁ <*> t₂ := sorry

theorem Coitree.seqRight_eq (t₁: Coitree E R₁) (t₂: Coitree E R₂):
    t₁ *> t₂ = Function.const R₁ id <$> t₁ <*> t₂ := sorry

theorem Coitree.pure_seq (f: R₁ → R₂) (t: Coitree E R₁):
    pure f <*> t = f <$> t := sorry

theorem Coitree.map_pure (f: R₁ → R₂) (r: R₁):
    f <$> (pure r) = pure (f := Coitree E) (f r) := sorry

theorem Coitree.seq_pure (f: Coitree E (R₁ → R₂)) (r: R₁):
    f <*> pure r = (fun h => h r) <$> f := sorry

theorem Coitree.seq_assoc (t₁: Coitree E R₁)
    (t₂: Coitree E (R₁ → R₂)) (t₃: Coitree E (R₂ → R₃)):
    t₃ <*> (t₂ <*> t₁) = ((@Function.comp R₁ R₂ R₃) <$> t₃) <*> t₂ <*> t₁ := by
  sorry

instance {E}: LawfulApplicative (Coitree E) where
  seqLeft_eq   := Coitree.seqLeft_eq
  seqRight_eq  := Coitree.seqRight_eq
  pure_seq     := Coitree.pure_seq
  map_pure     := Coitree.map_pure
  seq_pure     := Coitree.seq_pure
  seq_assoc    := Coitree.seq_assoc


theorem Coitree.bind_pure_comp (f: R₁ → R₂) (t: Coitree E R₁):
    bind t (fun r => pure (f r)) = f <$> t := sorry


theorem Coitree.bind_map (f: Coitree E (R₁ → R₂)) (t: Coitree E R₁):
    bind f (. <$> t) = f <*> t := sorry

theorem Coitree.pure_bind (r: R₁) (k: R₁ → Coitree E R₂):
    bind (pure r) k = k r := sorry

theorem Coitree.bind_assoc (t: Coitree E R₁)
    (k₁: R₁ → Coitree E R₂) (k₂: R₂ → Coitree E R₃):
    bind (bind t k₁) k₂ = bind t (fun x => bind (k₁ x) k₂) := sorry

instance {E}: LawfulMonad (Coitree E) where
  bind_pure_comp  := sorry
  bind_map        := sorry
  pure_bind       := sorry
  bind_assoc      := sorry
-/

end LawfulCoitree
