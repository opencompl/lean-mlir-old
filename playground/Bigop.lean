namespace Bigop

section Preliminaries

def Seq (α : Type) := List α

def BigBody (β α) :=  α × (β → β → β) × Bool × β

def applyBig (body : BigBody β α) (x : β) : β :=
   let (_, op, b, v) := body
   if b then op v x else x

def reducebig (idx : β) (r : Seq α) (body : α → BigBody β α) : β :=
   r.foldr (applyBig ∘ body) idx

def bigop := @reducebig

def iota : Nat → Nat → List Nat
   | m, 0   => []
   | m, n+1 => m :: iota (m+1) n

def index_iota (m n : Nat) := iota m (n - m)

class Enumerable (α : Type) where
   elems : List α

instance : Enumerable Bool where
   elems := [false, true]

instance [Enumerable α] [Enumerable β]: Enumerable (α × β) where
  elems := Enumerable.elems.bind fun a => Enumerable.elems.map fun b => (a, b)

def finElemsAux (n : Nat) : (i : Nat) → i < n → List (Fin n)
   | 0,   h => [⟨0, h⟩]
   | i+1, h => ⟨i+1, h⟩ :: finElemsAux n i (Nat.ltOfSuccLt h)

def finElems : (n : Nat) → List (Fin n)
   | 0     => []
   | (n+1) => finElemsAux (n+1) n (Nat.ltSuccSelf n)

instance : Enumerable (Fin n) where
  elems := (finElems n).reverse

instance : OfNat (Fin (Nat.succ n)) m where
  ofNat := Fin.ofNat m

end Preliminaries

-- Declare a new syntax category for "indexing" big operators
-- declare_syntax_cat
declare_syntax_cat index
syntax ident "<-" term : index
syntax ident "<-" term "|" term : index
-- end
syntax term:51 "≤" ident "<" term : index
syntax term:51 "≤" ident "<" term "|" term : index
-- Primitive notation for big operators
syntax "\\big_" "[" term "," term "]" "(" index ")" term : term

-- We define how to expand `\big` with the different kinds of index
macro_rules
  | `(\big_ [$op, $idx] ($i:ident <- $r) $F) => `(bigop $idx $r (fun $i => ($i, $op, true, $F)))
  | `(\big_ [$op, $idx] ($i:ident <- $r | $p) $F) => `(bigop $idx $r (fun $i => ($i, $op, $p, $F)))
  | `(\big_ [$op, $idx] ($lower:term ≤ $i:ident < $upper) $F) => `(bigop $idx (index_iota $lower $upper) (fun $i => ($i, $op, true, $F)))
  | `(\big_ [$op, $idx] ($lower:term ≤ $i:ident < $upper | $p) $F) => `(bigop $idx (index_iota $lower $upper) (fun $i => ($i, $op, $p, $F)))

-- Sum
macro "Σ" "(" idx:index ")" F:term : term =>
  `(\big_ [Add.add, 0] ($idx) $F)
-- end

-- We can already use `Σ` with the different kinds of index.
#check Σ (i <- [0, 2, 4] | i != 2) i
#check Σ (10 ≤ i < 20 | i != 5) i+1
#check Σ (10 ≤ i < 20) i+1

-- Define `Π`
-- Prod
macro "Π" "(" idx:index ")" F:term : term =>
  `(\big_ [Mul.mul, 1] ($idx) $F)
-- end

-- The examples above now also work for `Π`
#check Π (i <- [0, 2, 4] | i != 2) i
#check Π (10 ≤ i < 20 | i != 5) i+1
#check Π (10 ≤ i < 20) i+1

-- We can extend our grammar for the syntax category `index`.
syntax ident "|" term : index
syntax ident ":" term : index
syntax ident ":" term "|" term : index
-- Add new rules
macro_rules
  | `(\big_ [$op, $idx] ($i:ident : $type) $F) => `(bigop $idx (Enumerable.elems (α := $type)) (fun $i => ($i, $op, true, $F)))
  | `(\big_ [$op, $idx] ($i:ident : $type | $p) $F) => `(bigop $idx (Enumerable.elems (α := $type)) (fun $i => ($i, $op, $p, $F)))
  | `(\big_ [$op, $idx] ($i:ident | $p) $F) => `(bigop $idx Enumerable.elems (fun $i => ($i, $op, $p, $F)))

-- The new syntax is immediately available for all big operators that we have defined
def myPred (x : Fin 10) : Bool := true
#check Σ (i : Fin 10) i+1
#check Σ (i : Fin 10 | i != 2) i+1
#check Σ (i | myPred i) i+i
#check Π (i : Fin 10) i+1
#check Π (i : Fin 10 | i != 2) i+1
#check Π (i | myPred i) i+i

-- We can easily create alternative syntax for any big operator.
macro "Σ" idx:index "," F:term : term =>
  `(Σ ($idx) $F)

#check Σ 10 ≤ i < 20, i+1

-- Finally, we create a command for automating the generation of big operators.
syntax "def_bigop" str term:max term:max : command
-- Antiquotations can be nested as in `$$F`, which expands to `$F`, i.e. the value of
-- `F` is inserted only in the second expansion, the expansion of the new macro `$head`.
macro_rules
  | `(def_bigop $head $op $unit) =>
    `(macro $head:strLit "(" idx:index ")" F:term : term => `(\big_ [$op, $unit] ($$idx) $$F))

def_bigop "SUM" Nat.add 0
#check SUM (i <- [0, 1, 2]) i+1

end Bigop
