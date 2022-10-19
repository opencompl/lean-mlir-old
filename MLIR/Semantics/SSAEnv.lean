/-
## SSA environment

This file implements the SSA environment which maps variables names from
different scopes to explicitly-typed values. It is, conceptually, a map
`SSAVal → (α: MLIRType δ) × α` for each scope.

The main concern with such state is the definition, maintainance and use of the
property that the values are defined only once, allowing us to study sections
of the code while skipping/omitting operations very freely. Bundling uniqueness
invariants with the data structure would result in very tedious and context-
dependent maintenance, which is not a friendly option.

Instead, we ignore the SSA constraints when defining semantics and interpreting
programs, assuming the language allows shadowing and overriding values (of
course, valid programs won't do that). We only define SSA constraints later on
to prove that transformations are context-independent.

`SSAScope` implements a single scope as a list of name/value pairs and supports
edition.

`SSAEnv` implements a stack of scopes. New scopes are created when entering
regions. Here again nothing prevents a region from shadowing variables or
accessing out-of-scope values (in the case of an isolated region), but we only
care when proving the correction of transformations.
-/

import MLIR.Semantics.Fitree
import MLIR.Semantics.Types
import MLIR.Util.WriterT
import MLIR.Util.Tactics

import MLIR.AST
open MLIR.AST

section

def SSAScope (t: Type) [ct: Code t] :=
  List (SSAVal × (τ: MLIRType t) × τ.eval)

@[simp]
def SSAScope.getT [ct: Code t] (name: SSAVal):
  SSAScope t → Option ((τ: MLIRType t) × τ.eval)
  | [] => none
  | ⟨name', τ, v⟩ :: l =>
      if name' = name then some ⟨τ,v⟩ else getT name l

@[simp]
def SSAScope.get [ct: Code t] (name: SSAVal):
  SSAScope t → (τ: MLIRType t) → Option τ.eval
  | [], _ => none
  | ⟨name', τ', v'⟩ :: l, τ =>
      if H: name' = name then
        if H': τ' = τ then
          some (cast (by simp [H']) v')
        else
          none
      else get name l τ

@[simp]
def SSAScope.set [ct: Code t] (name: SSAVal) (τ: MLIRType t) (v: τ.eval):
  SSAScope t → SSAScope t
  | [] => [⟨name, τ, v⟩]
  | ⟨name', τ', v'⟩ :: l =>
      if name' = name
      then ⟨name', τ, v⟩ :: l
      else ⟨name', τ', v'⟩ :: set name τ v l

def SSAScope.str [ct: Code t] (scope: SSAScope t): String :=
  "\n".intercalate <| scope.map fun ⟨name, τ, v⟩ => s!"{name} = {v} : {τ}"

instance [ct: Code t]: ToString (SSAScope t) where
  toString := SSAScope.str

/- Maybe useful in the future, for proofs
def SSAScope.has (name: SSAVal) (l: SSAScope): Bool :=
  l.any (fun ⟨name', _, _⟩ => name' == name)

def SSAScope.free (name: SSAVal) (l: SSAScope): Bool :=
  l.all (fun ⟨name', _, _⟩ => name' != name)

def SSAScope.maps (l: SSAScope) (name: SSAVal) (τ: MLIRTy) (v: τ.eval) :=
  l.Mem ⟨name, τ, v⟩ -/

-- SSAScope proofs

theorem SSAScope.getT_set_ne [ct: Code t] (v v': SSAVal):
    v' ≠ v →
    ∀ (scope: SSAScope t) (τ: MLIRType t) val,
    getT v (set v' τ val scope) = getT v scope := by
  intros Hne scope τ val
  induction scope with
  | nil => simp [Hne]
  | cons head tail =>
    simp
    byCases H: head.fst = v'
    . simp [Hne]
    . byCases H2: head.fst = v
      assumption

theorem SSAScope.getT_set_eq [ct: Code t] (scope: SSAScope t) (v: SSAVal) (τ: MLIRType t) val:
    getT  v (set  v τ val scope) = some ⟨τ, val⟩  := by
  induction scope with
  | nil => simp
  | cons head tail =>
    simp
    byCases H: head.fst = v
    assumption

theorem SSAScope.get_set_ne_val [ct: Code t] (v v': SSAVal):
    v' ≠ v →
    ∀ (scope: SSAScope t) (τ τ': MLIRType t) val,
    get v (set v' τ val scope) τ' = get v scope τ' := by
  intros Hne scope τ τ' val
  induction scope with
  | nil => simp [Hne]
  | cons head nil =>
    simp
    byCases H: head.fst = v'
    . simp [Hne]
    . byCases H2: head.fst = v <;> try assumption

theorem SSAScope.get_set_ne_type [ct: Code t] (τ τ': MLIRType t):
    τ' ≠ τ →
    ∀ (scope: SSAScope t) (v: SSAVal) val,
    get v (set v τ' val scope) τ = none := by
  intros Hne scope v val
  induction scope with
  | nil => simp [Hne]
  | cons head tail Hind =>
    simp
    byCases H: head.fst = v
    . simp [Hne]
    . byCases H2: head.fst = v <;> try assumption

theorem SSAScope.get_set_eq [ct: Code t] (v: SSAVal) (scope: SSAScope t) (τ: MLIRType t) val:
    get v (set v τ val scope) τ = some val := by
  induction scope with
  | nil => simp; apply cast_eq
  | cons head nil =>
    simp
    byCases H: head.fst = v <;> try apply cast_eq
    assumption

-- SSAEnv

inductive SSAEnv (t: Type) [ct: Code t] :=
  | One (scope: SSAScope t)
  | Cons (head: SSAScope t) (tail: SSAEnv t)

instance [Code t]: Inhabited (SSAEnv t) where
  default := .One []

-- An SSA environment with a single empty SSAScope
def SSAEnv.empty [ct: Code t]: SSAEnv t := One []

def SSAEnv.str [ct: Code t] (env: SSAEnv t): String :=
  match env with
  | One s => s.toString
  | Cons head tail => head.toString ++ "---\n" ++ tail.str

instance {ct: Code t}: ToString (SSAEnv t) where
  toString := SSAEnv.str

def SSAEnv.getT [ct: Code t] (name: SSAVal):
  SSAEnv t → Option ((τ: MLIRType t) × τ.eval)
  | One s => s.getT name
  | Cons s l => s.getT name <|> getT name l

def SSAEnv.get [ct: Code t] (name: SSAVal) (τ: MLIRType t):
  SSAEnv t → Option τ.eval
  | One s => s.get name τ
  | Cons s l => s.get name τ <|> get name τ l

@[simp] def SSAEnv.get_One [ct: Code t] {scope: SSAScope t (ct := ct)} {τ: MLIRType t}:
  SSAEnv.get name τ (.One scope) = scope.get name τ := rfl

def SSAEnv.set [ct: Code t] (name: SSAVal) (τ: MLIRType t) (v: τ.eval):
  SSAEnv t → SSAEnv t
  | One s => One (s.set name τ v)
  | Cons s l => Cons (s.set name τ v) l

@[simp] def SSAEnv.set_One [ct: Code t] {scope: SSAScope t (ct := ct)} {τ: MLIRType t} {v: τ.eval}:
  SSAEnv.set name τ v (.One scope) = .One (scope.set name τ v) := rfl

instance {ct: Code t}: DecidableEq ((τ: MLIRType t) × τ.eval) :=
  fun ⟨τ₁, v₁⟩ ⟨τ₂, v₂⟩ =>
    if H: τ₁ = τ₂ then
      if H': cast (by simp [H]) v₁ = v₂ then
        isTrue (by cases H; cases H'; simp [cast_eq])
      else isFalse fun h => by cases h; cases H' rfl
    else isFalse fun h => by cases h; cases H rfl

def SSAEnv.eqOn [ct: Code t] (l: List SSAVal) (env₁ env₂: SSAEnv t): Bool :=
  l.all (fun v => env₁.getT v == env₂.getT v)

-- SSAEnv theorems

theorem SSAEnv.getT_set_ne [ct: Code t] (v v': SSAVal):
    v' ≠ v →
    ∀ (env: SSAEnv t) (τ: MLIRType t) val,
    getT v (set v' τ val env) = getT v env := by
  intros Hne env τ val
  cases env with
  | One s =>
    simp [getT, set]
    rw [SSAScope.getT_set_ne]
    assumption
  | Cons head tail =>
    simp [getT, set, HOrElse.hOrElse, OrElse.orElse, Option.orElse]
    rw [SSAScope.getT_set_ne]
    assumption

theorem SSAEnv.getT_set_eq [ct: Code type] (env: SSAEnv type) (v: SSAVal) (τ: MLIRType type) val:
    getT v (SSAEnv.set v τ val env) = some ⟨τ, val⟩  := by
  cases env with
  | One s =>
    simp [getT, set]
    rw [SSAScope.getT_set_eq]
  | Cons head tail =>
    simp [getT, set, HOrElse.hOrElse, OrElse.orElse, Option.orElse]
    simp [SSAScope.getT_set_eq]

theorem SSAEnv.get_set_ne_val [ct: Code t] (v v': SSAVal):
    v' ≠ v →
    ∀ (env: SSAEnv t) (τ τ': MLIRType t) val,
    get v τ' (set v' τ val env) = get v τ' env := by
  intros Hne env τ τ' val
  cases env with
  | One s =>
    simp [get, set]
    rw [SSAScope.get_set_ne_val]
    assumption
  | Cons head tail =>
    simp [get, set, HOrElse.hOrElse, OrElse.orElse, Option.orElse]
    rw [SSAScope.get_set_ne_val]
    assumption

theorem SSAEnv.get_set_eq [ct: Code type] (v: SSAVal) (env: SSAEnv type) (τ: MLIRType type) val:
    get v τ (set v τ val env) = some val := by
  cases env with
  | One s =>
    simp [get, set]
    rw [SSAScope.get_set_eq]
  | Cons head tail =>
    simp [get, set, HOrElse.hOrElse, OrElse.orElse, Option.orElse]
    rw [SSAScope.get_set_eq]


-- Interactions manipulating the environment

inductive SSAEnvE (type: Type) [ct: Code type]: Type → Type where
  | Get: (τ: MLIRType type) → SSAVal → SSAEnvE type τ.eval
  | Set: (τ: MLIRType type) → SSAVal → τ.eval → SSAEnvE type Unit

@[simp_itree]
def SSAEnvE.handle {E} {type: Type} [ct: Code type]:
  SSAEnvE type ~> StateT (SSAEnv type) (Fitree E) :=
  fun _ e env =>
    match e with
    | Get τ name =>
        match env.get name τ with
        | some v => return (v, env)
        | none => return (default, env)
    | Set τ name v =>
        return (.unit, env.set name τ v)

def SSAEnvE.handleLogged {E} {type: Type} [ct: Code type]:
    SSAEnvE type ~> WriterT (StateT (SSAEnv type) (Fitree E)) :=
  fun _ e => do
    let env <- WriterT.lift StateT.get
    match e with
    | Get τ name =>
        match env.get name τ with
        | some v => do
            logWriterT s!"get {name} (={v}); "
            return v
        | none =>
            logWriterT s!"get {name} (not found!); "
            return default
    | Set τ name v =>
        logWriterT s!"set {name}={v}; "
        WriterT.lift $ StateT.set (env.set name τ v)
        return ()

@[simp_itree]
def SSAEnv.get? {E} [ct: Code type] [Member (SSAEnvE type) E]
  (τ: MLIRType type) (name: SSAVal): Fitree E τ.eval :=
    Fitree.trigger (SSAEnvE.Get τ name)

@[simp_itree]
def SSAEnv.set? {E} [ct: Code type] [Member (SSAEnvE type) E]
    (τ: MLIRType type) (name?: Option SSAVal) (v: τ.eval): Fitree E Unit :=
  match name? with
  | some name =>
      Fitree.trigger (SSAEnvE.Set τ name v)
  | none =>
      return ()

-- Handlers

def interpSSA [ct: Code type] (t: Fitree (SSAEnvE type) R): StateT (SSAEnv type) (Fitree Void1) R :=
  t.interpState SSAEnvE.handle

def interpSSA'  [ct: Code type] {E} (t: Fitree (SSAEnvE type +' E) R):
    StateT (SSAEnv type) (Fitree E) R :=
  t.interpState (Fitree.case SSAEnvE.handle Fitree.liftHandler)

def interpSSALogged [ct: Code type] (t: Fitree (SSAEnvE type) R):
    WriterT (StateT (SSAEnv type) (Fitree Void1)) R :=
  t.interp SSAEnvE.handleLogged

def interpSSALogged' [ct: Code type]  {E} (t: Fitree (SSAEnvE type +' E) R):
    WriterT (StateT (SSAEnv type) (Fitree E)) R :=
  t.interp (Fitree.case SSAEnvE.handleLogged Fitree.liftHandler)

@[simp] theorem interpSSA'_Vis_left [ct: Code type]
    (k: T → Fitree (SSAEnvE type +' E) R) (e: SSAEnvE type T) (s₁: SSAEnv type):
  interpSSA' (Fitree.Vis (Sum.inl e) k) s₁ =
  Fitree.bind (SSAEnvE.handle _ e s₁) (fun (x,s₂) => interpSSA' (k x) s₂) :=
  rfl

@[simp] theorem interpSSA'_Vis_right [Code type] (k: T → Fitree (SSAEnvE type +' E) R):
  interpSSA' (Fitree.Vis (Sum.inr e) k) =
  fun s => Fitree.Vis e (fun x => interpSSA' (k x) s) := rfl

@[simp] theorem interpSSA'_ret [ct: Code type]: interpSSA' (ct := ct) (E := E) (Fitree.ret r) = fun s => Fitree.ret (r,s) := rfl

private theorem pair_eta {α β: Type} (x: α × β): (x.fst, x.snd) = x :=
  match x with
  | (_, _) => rfl

@[simp] theorem interpSSA'_trigger_MemberSumL [Code type]
    (e: SSAEnvE type T) (s₁: SSAEnv type):
  interpSSA' (@Fitree.trigger (SSAEnvE type) (SSAEnvE type +' E) _ MemberSumL e) s₁ =
  SSAEnvE.handle _ e s₁ := by
  simp [Fitree.trigger, pair_eta]

theorem interpSSA'_bind [ct: Code type]
    (t: Fitree (SSAEnvE type +' E) T) (k: T → Fitree (SSAEnvE type +' E) R)
    (s₁: SSAEnv type):
  interpSSA' (type := type) (Fitree.bind t k) s₁ =
  Fitree.bind (interpSSA' t s₁) (fun (x,s₂) => interpSSA' (k x) s₂) := by
  apply Fitree.interpState_bind
