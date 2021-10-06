-- Hoare monad: https://www.microsoft.com/en-us/research/wp-content/uploads/2013/06/paper-pldi13.pdf
-- 2021 Coq formalization + iTrees: https://www.cis.upenn.edu/~stevez/papers/SZ21.pdf
-- Principles of program verification for arbitrary monadic effects: https://hal.archives-ouvertes.fr/tel-02416788/document


-- | heap 
inductive Heap (k: Type) (v: Type): Type where
| Empty: Heap k v
| Set: k -> v -> Heap k v -> Heap k v

def get  {k v: Type} [BEq k] (h: Heap k v)  (needle: k): Option v :=
 match  h with
 | Heap.Empty => none
 | Heap.Set needle' v h' => if needle == needle' then some v else get h' needle

def set (h: Heap k v) (needle: k) (value: v): Heap k v := 
  Heap.Set needle value h

-- | state monad
inductive State (k: Type) (v: Type) (a: Type): Type where
| mk: (Heap k v -> (a × Heap k v)) -> State k v a

def runState  (s: State k v a) (h: Heap k v): a × Heap k v :=
  match s with
  | State.mk f => f h

def statePure (value: a): State k v a :=
  State.mk (fun h => (value, h))

def stateBind (sa: State k v a) (a2sb: a -> State k v b): State k v b :=
    State.mk (fun h => let (a, h') := runState sa h
                       let s' := a2sb a
                       runState s' h')

instance : Monad (State k v) := {
  pure := statePure,
  bind := stateBind
}


inductive Pred (a: Type) := 
| mk: (a -> Prop) -> Pred a

def runPred (pred: Pred a) (v: a): Prop :=
  match pred with 
  | Pred.mk p => p v

def predPure (value: a): Pred a := Pred.mk fun v => v = value
def predBind (pa: Pred a) (a2pb: a -> Pred b): Pred b :=
  Pred.mk $ fun b => ∃ (aval: a), runPred pa aval ∧ runPred (a2pb aval) b

-- https://hal.archives-ouvertes.fr/tel-02416788/document
instance : Monad Pred := {
  pure := predPure,
  bind := predBind
}


-- https://hal.archives-ouvertes.fr/tel-02416788/document
structure PrePost (t: Type) where
  pre: Prop
  post: t -> Prop

def prepostPure (a: t): PrePost t := 
  { pre := True, post := fun b => a = b }

def prepostBind (ma: PrePost t) (a2mb: t -> PrePost t'): PrePost t' :=
 { pre := ma.pre ∧ ∀ (a: t), ma.post a -> (a2mb a).pre,
   post := fun b => ∃ (a: t), ma.post a ∧ (a2mb a).post b
 }

instance: Monad PrePost := {
  pure := prepostPure,
  bind := prepostBind
}



-- instance : Monad (State k v) where
--   pure value := State.mk  (fun h => (value, h))
--   bind sa a2sb := 
--     State.mk (fun h => let (a, h') := runState sa h
--                        let s' := a2sb a
--                        runState s' h')

-- | weakest precondition, obtained from * over State, as described
-- at https://arxiv.org/pdf/1608.06499.pdf
inductive WeakPre (k: Type) (v: Type) (a: Type): Prop where
| mk: ((Heap k v) -> ((a × Heap k v) -> Type) -> Type) -> WeakPre k v a


def weakprePure (value: a): WeakPre k v a :=
  WeakPre.mk $ fun s0 post => post (value, s0)


-- | djikstra monad in all its abstract glory, courtesy steve
inductive Djik
  (M: Type u -> Type u)
  (W: Type u -> Type u)
  (obs: forall {t: Type u}, M t -> W t)
  {A: Type u} [LE (W A)] (w: W A) where
| mk: (m: M A) -> (p: obs m <= w) -> Djik M W obs w




-- | slice category of propopositions over pre.
-- | propositions which imply q
inductive PropImplying (q: Prop): Prop where
| mk:(prop: Prop) -> (imply: prop -> q) -> PropImplying q

-- structure FwdPred (t: Type) where
--   pred: (pre: Prop) -> (t -> PropImplying pre)
--   monotone: 

