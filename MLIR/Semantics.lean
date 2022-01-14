import Std.Data.RBMap
-- https://github.com/leanprover/lean4/blob/master/src/Std/Data/AssocList.lean#L10
open Std

-- Djikstra monads to define semantics.
-- Plus convenient helpers for 'default' things.
-- open question: how does this work with, like, unbounded computations like loop?
-- do we restrict to a subset where we have termination bounds?
-- do we use djikstra monads forever?

instance [ToString α ] [ToString β] {compare: α -> α -> Ordering}: ToString (RBMap α β compare) where
  toString (x: RBMap α β compare) := 
    toString (x.toList)

instance [BEq α] [BEq β] {compare: α -> α -> Ordering}: BEq (RBMap α β compare) where
  beq (x y: RBMap α β compare) := 
    BEq.beq (x.toList) (y.toList) -- TODO: this assumes they are made into lists based on ordering!


abbrev Set (α : Type) (compare: α -> α -> Ordering) := RBMap α Unit compare


def compareTuple [Ord α] [Ord β] (p: α × β) (q: α × β)  : Ordering := 
            match compare p.fst q.fst with
            | Ordering.lt => Ordering.lt
            | Ordering.gt => Ordering.gt
            | Ordering.eq => compare p.snd q.snd 

instance [Ord α] [Ord β] : Ord (α × β) where
    compare := compareTuple


def RBMap.union_set {α: Type} {compare:α -> α -> Ordering} (xs: Set α compare) (ys: Set α compare): Set α compare := 
  RBMap.fromList (xs.toList ++ ys.toList) compare

def RBMap.set_bind {α: Type} {compare: α -> α -> Ordering} (xs: Set α compare) (f: α -> Set α compare): Set α compare :=
  RBMap.fold (fun new a () => RBMap.union_set new (f a)) RBMap.empty xs 

def RBMap.set_map {α: Type} {compare: α -> α -> Ordering} (xs: Set α compare) (f: α -> β) {compare': β -> β -> Ordering}: Set β compare' := 
    RBMap.fold (fun new a () => RBMap.insert new (f a) ()) RBMap.empty xs

def RBMap.set_subtract {α: Type} {compare: α -> α -> Ordering} (all: Set α compare) (to_remove: Set α compare): Set α compare :=
  RBMap.fold (fun all' a () => RBMap.erase all' a) to_remove all 

def RBMap.set_singleton {α: Type} (a: α) (compare: α -> α -> Ordering): Set α compare :=
    RBMap.fromList [(a, ())] compare

def RBMap.set_empty {α: Type} (compare: α -> α -> Ordering): Set α compare :=
    RBMap.empty

def RBMap.set_from_list {α: Type} (as: List α) (compare: α -> α -> Ordering): Set α compare :=
    RBMap.fromList (as.map (fun a => (a, ()))) compare

def RBMap.set_insert {α: Type} {compare: α -> α -> Ordering} (as: Set α compare) (a: α): Set α compare :=
    RBMap.insert as a ()

def RBMap.set_union {α: Type} {compare: α -> α -> Ordering} (s1 s2: Set α compare): Set α compare := 
   s1.fold (fun out k () => RBMap.set_insert out k) s2

-- partial def randSetM (minSize: Nat) (maxSize: Nat) (ra: Rand α)  {compare: α -> α -> Ordering}:
--     Rand (Set α compare) := do 
--    let size <- randIntM minSize maxSize
--    let rec go (s: Set α compare) := 
--     if s.size == size
--     then return s
--     else do 
--         let a <- ra
--         go (RBMap.set_insert s a) 
--     go (RBMap.set_empty compare)

-- 
def RBMap.union_keep_right {α: Type} {compare: α -> α -> Ordering}
    (s1 s2: RBMap α  β compare): RBMap α  β compare := 
   s1.fold (fun out k v => out.insert k v) s2

def RBMap.set_to_list {α: Type} {compare: α -> α -> Ordering} (as: Set α compare): List α := 
    (RBMap.toList as).map (fun (a, ()) => a)

-- filter seems expensive?
def RBMap.set_filter {α: Type} (p: α → Bool)
    {compare: α -> α -> Ordering} (as: Set α compare): Set α compare := Id.run do
    let mut out := RBMap.set_empty compare
    for (a, ()) in as do
        if p a then
        out := RBMap.insert out a ()
    return out


-- inductive RunCombExpr : RBMap String Int compare -> Op -> RBMap String Int -> Prop where
-- | RunFoo: ∀ (m: RBMap String Int compare) (o: op), RunCombExpr m o m
