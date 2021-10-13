inductive op where
| add: String -> op -> op
| done: op

inductive is_done : op -> Prop where
  | is_done_done: is_done op.done
  | is_done_add: (s: String) -> (x: op) -> is_done x -> is_done (op.add s x)

-- | do case analysis on PRF, show that it must have been created by is_done_add.
-- TODO: how do I use `inversion`?
def is_done_step_bwd (s: String) (o: op) (PRF: is_done (op.add s o)): is_done o  := 
match PRF with | is_done.is_done_add _ _ prf => prf


def is_done_step (s: String) (o: op) (PRF: is_done o): is_done (op.add s o) := 
  is_done.is_done_add s o PRF
-- | build stepwise.
-- TODO: this lives in prop, so you cannot eliminate into Type.
-- fuck large elimination x(


def proof : ∃ o, is_done o := by {
  apply Exists.intro;
  apply is_done_step_bwd "foo";
  apply is_done_step_bwd "bar";
  apply is_done.is_done_add;
  apply is_done.is_done_add;
  apply is_done.is_done_done;
}



  
    
   
def main: IO Unit :=
  IO.println "foo"



