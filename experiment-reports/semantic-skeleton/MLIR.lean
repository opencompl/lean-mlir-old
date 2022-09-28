import Lean
open Std
open Lean
/-
Terminating fragment of lambda calculus (no fixpoint combinator, assume you are not perverse
 and write your own (eg. assume this is STLC or some such.
-/
inductive Lam where
| App: Lam -> Lam -> Lam
| Var: Nat -> Lam -- numnber of parents to be walked up.
| Lam: Lam -> Lam 
-- | Eval: Lam -> Lam
 deriving BEq, Inhabited


 instance : Coe Nat Lam where
   coe := .Var
  
 instance : OfNat Lam n where
   ofNat := .Var n



-- (fun x => fun y => fun z => x y z)
-- (fun => fun => fun => 2 1 0)

-- (fun x => fun y => fun z => x y z) (fun k => k)
-- (fun => fun => fun => 2 1 0) (fun 0 => 0)
-- (fun => fun => 1 0 (fun 0 => 0))

-- (fun x => x $ fun y => x $ fun z => x y z) (fun k => k)
-- (fun => 0 $ fun => 0 $ 1 $ fun => 2 1 0) (fun 0 => 0)
-- (fun => 0 $ (fun  => 0) $ fun => 1 0 (fun 0 => 0))

-- (fun 0 => (fun0 => fun 1 => 0 1 1) 0 0) 
-- body[var = val]
@[simp]
def subst (body: Lam) (val: Lam)  (onePlusLargestLegal: Nat := 0): Lam  := 
  match body with 
  | .App f x => .App (subst f val onePlusLargestLegal) (subst x val onePlusLargestLegal)
  | .Var n => if n < onePlusLargestLegal
    then .Var n else if n == onePlusLargestLegal then val else panic! "unexpected binder ix {n}"
  | .Lam body => subst body val (onePlusLargestLegal + 1)

abbrev Env := HashMap String Lam
abbrev EvalM α := StateM Env α 

/-
A toplevel machine, whose equational theory captures non-computability.
Lambda calculus is annoying, needs evaluation contexts.
-/
inductive Oracle: Type where
| Running: Env -> Lam -> Oracle -- taking a step
| GroundTerm: Env -> Lam -> Oracle -- stuck term.

def Oracle.project: Oracle -> Env × Lam
| .Running e l => (e, l)
| .GroundTerm e l => (e, l)

@[reducible, simp]
def step_ (t: Lam): Lam :=
 match t with 
 | .App rator rand => 
        match rator with 
        | .Lam body => subst body rand
        | _ => .App (step_ rator) rand
 | .Lam body => .Lam (step_ body)
 | .Var v => .Var v

@[reducible, simp]
 def step (e: Env) (t: Lam): Oracle := .Running e (step_ t)



-- y 
-- ((fun g f => f (g f)) (fun g f => f (g f))) (fun x => x)

-- let y f = f (y f)

-- s f g x = f x (g x)
@[reducible, simp]
def s : Lam := (.Lam ( .Lam ( .Lam (.App (.App 2 0) (.App 1 0)))))
@[reducible, simp]
def k : Lam := (.Lam (.Lam 1))

@[reducible, simp]
def id_ : Lam := (.Lam 0)
@[reducible, simp]
def skkid : Lam := (.App (.App (.App s k) k) id_)
@[reducible, simp]
def idid : Lam := (.App id_ id_)
-- @[reducible, simp]
-- def y_id : Lam := ((.Lam "g" (.Lam "f"  ("f" $$ ("g" $$ "f")))) $$  (.Lam "k" (.Lam "l"  ("l" $$ ("k" $$ "l"))))) $$ (.Lam "x" "x")


namespace Eqn
 axiom compute_oracle: Oracle -> Oracle
 axiom compute_oracle_running (env: Env) (lam: Lam): compute_oracle (.Running env lam) = step env lam

theorem step_id_id: compute_oracle (Oracle.Running {} idid) = compute_oracle (Oracle.Running {} id_) := by {
  rewrite [compute_oracle_running]; 
  simp; 
  rewrite [compute_oracle_running]; 
  simp;
}

theorem step_skkid: compute_oracle (Oracle.Running {} skkid) = compute_oracle (Oracle.Running {} id_) := by {
  rewrite [compute_oracle_running]; 
  simp; 
  simp; 
  rewrite [compute_oracle_running]; 
  simp; 
  simp; 
  simp; 

}

end Eqn


namespace Quotient
 inductive R: Oracle -> Oracle -> Prop where
 | Run: R (.Running env lam) (step env lam)

 -- quotient of oracle by run rules
 abbrev Oracle' := Quot.mk R
 #check Oracle'

theorem step_omega: ∃ (e: Env), Oracle' (Oracle.Running {} skkid) = Oracle' (Oracle.Running e id_) := by {
  simp[R.Run];
  simp[R.Run];
}
end Quotient
