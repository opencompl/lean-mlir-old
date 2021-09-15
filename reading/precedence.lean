structure LHS where
  name: String


structure Binding where
  lhs: LHS
  rhs: Int

structure V where
  v: Int

structure W where
  v1: V
  v2: V

prefix:20 "&" => V.mk
prefix:30 "&&" => V.mk
prefix:40 "&&&" => V.mk
infix:10 "===" => W.mk
infix:20 "=&=" => W.mk
infix:30 "=&&=" => W.mk
infix:40 "=&&&=" => W.mk

syntax:19  term:19 ":=&&&=:" term:19 : term
macro_rules  | `($a :=&&&=: $b) => `(W.mk $a $b)


prefix:300 "%" => LHS.mk
infix:100 "=:=" => Binding.mk

#check (% "x") =:= 10
#check % "x" =:= 10


syntax:30  term "##" term : term
macro_rules  | `($a ## $b) => `($a + $b)

syntax:30  term:30 "!!" term:30 : term
macro_rules | `($a !! $b) => `($a + $b)

syntax:30 term:30 "@@" term:31 : term
macro_rules | `($a @@ $b) => `($a + $b)

--  | I don't undestand this.
syntax:30 term:31 "$$" term:30 : term
macro_rules | `($a $$ $b) => `($a + $b)

syntax:31 term:30 "xxx" term:30 : term
macro_rules  | `($a xxx $b) => `($a + $b)

syntax:30 term:31 "yyy" term:31 : term
macro_rules  | `($a yyy $b) => `($a + $b)


#check (1 ## 2 ## 3 ## 4)
#check (1 !! 2 !! 3 !! 4)
#check (1 @@ 2 @@ 3 @@ 4)
#check (1 $$ 2 $$ 3 $$ 4)

#check (1 xxx 2 xxx 3 xxx 4)
#check (1 yyy 2)
-- | this does not parse, because parser is upward closed filter. We expect a term of precedence AT LEAST what's marked
-- think of ":" as ">=": ie, I will parse if the precedence is ">= threshold".
-- #check (1 yyy 2 yyy 3)



#check (&20)
#check (&&20)
#check (&&&20)
-- error because infix and prefix have same fixity?
-- #check (&20 =:= &20)


#check (&20 === &20)
-- cannot parse because infix has higher precedence?
-- #check (&20 =&&&= &20)

-- Mental model of parser is that the precedence level is an upward filter.
-- Only allows to glob terms of higher precedence.

-- This is fine:
-- =============
-- prefix:20 "&" => V.mk
-- syntax:19  term ":=&&&=:" term:19 : term
-- This is fucked:
-- ==============
-- prefix:20 "&" => V.mk
-- syntax:20  term ":=&&&=:" term:19 : term
--       ^^^-- CHANGED
-- because the precedence of the infix expression should be LESS than the precedence of the prefix
-- to "break the parsing" so it backtracks in an attempt to find 
#check (&20 :=&&&=: &20)
