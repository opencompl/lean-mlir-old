inductive Foo
| mk: Int -> Foo

instance : Inhabited Foo where
  default := Foo.mk 42

constant odd : Int -> Foo
constant even : Int -> Foo

-- | in reality these are partial mutually defined functions.
-- | We use `constant` to evade
-- | `partial mutual...end`
mutual
  def even_impl (i: Int): Foo := 
        match i with 
        | 0 => Foo.mk 1
        | n => even (n - 1)
  
 def odd_impl (i: Int): Foo := 
     match i with 
     | 0 => Foo.mk 0
     | n => even (n - 1)
end


attribute [implementedBy odd_impl] odd
attribute [implementedBy even_impl] even




-- | expected output: 1
def main : IO Unit := 
  match even 10 with
   | Foo.mk k => IO.println k
    
