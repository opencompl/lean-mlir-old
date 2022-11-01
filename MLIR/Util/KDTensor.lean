import MLIR.Util.Mathlib4.NatBasic
import MLIR.Util.Mathlib4.Dvd
import MLIR.Util.Mathlib4.NatLemmas
import MLIR.Util.List
import MLIR.Util.FinInt

/-
This file defines the theory of K-dimensional arrays (for fixed K=4).
This aspires to be generalized to arbitrary dimensions, but for now,
we develop the theory for fixed dimension.

TODO: please unify:
- MLIR/Model/BuiltinModel.lean
- MLIR/Util/KDTensor.lean
- MLIR/Semantics/TensorElem.lean
-/

/-
#### Finite Integers, Finite Domains, Endomorphisms, Permutations
-/

-- function with finite domain.
@[simp, reducible]
abbrev Findom (n: Nat) (α: Type) := Fin n → α

def Findom.nil: Findom 0 α := fun ix => ix.elim0


def castFindom (n m : Nat) (EQ: n = m) (v: Findom n α) : Findom m α :=
  fun ix => v (EQ ▸ ix)

-- Nil is uniquely inhabited, as there is only one function (0 -> α)
-- upto extensionality.
theorem Findom.nil_unique: ∀ (f: Findom 0 α), f = Findom.nil := by {
  intros f
  simp[Findom, Findom.nil] at *;
  funext x;
  apply x.elim0;
}


-- increment a fin to live in a larger universe.
def Fin.increment (f: Fin n): Fin (Nat.succ n) :=
  { val := Nat.succ f.val, isLt := by { have H : _ := f.isLt; simp_arith at *; apply H; } }


-- any natural than one is zero.
@[simp]
theorem Nat.zero_of_lt_one (n: Nat) (LTONE: n < 1): n = 0 := by {
  cases LTONE;
  case refl =>  { simp; }
  case step H => { simp_arith [H]; contradiction; }
}

theorem Nat.lt_one_is_zero (n: Nat) (LTONE: n < 1): n = 0 := Nat.zero_of_lt_one n LTONE

theorem Fin.one_unique: ∀ (f: Fin 1), f = 0 := by {
  intros f;
  cases f;
  case mk val isLt => {
    have H: val = 0 := by {
      apply Nat.lt_one_is_zero <;> assumption;
    }
    simp[H];
    congr;
  }
}


-- get the last element of a fin.
def Fin.last (n: Nat): Fin (Nat.succ n) :=
  match n with
  | 0 => 0
  | Nat.succ n' => ⟨Nat.succ n', by { simp_arith; }⟩

-- enlarge the 'n' to 'n+1' of Fin.
def Fin.lift (f: Fin n): Fin (Nat.succ n) :=
  { val := f.val, isLt := by { have H : f.val < n := f.isLt; apply Nat.lt_of_lt_of_le; exact H; simp_arith; } }

-- lift(?x : Fin n) != last (n+1)
-- lift does not have last in its codomain.
theorem Fin.lift_neq_last_succ: ∀ (x: Fin (Nat.succ n)), (Fin.lift x) ≠ Fin.last (Nat.succ n) := by {
  intros x;
  simp[Fin.last, Fin.lift];
  have X : x.val < Nat.succ n := x.isLt;
  simp_arith;
  by_contra CONTRA;
  simp [CONTRA] at X;
}


@[simp]
theorem le_of_le_of_neq_upper_bound {x N : ℕ} (LEQ: x ≤ N) (NEQ: x ≠ N): x < N := by sorry

-- if not last, then value is less than n
theorem Fin.lt_n_of_not_last (f: Fin (Nat.succ n)) (NOTLAST: f ≠ (Fin.last n)): f.val < n := by{
  cases f;
  case mk v isLt => {
    simp[last] at NOTLAST;
    cases n;
    case zero => {
      simp[NOTLAST];
      simp at NOTLAST;
      -- v < 1 => v = 0
      have H : v = 0 := by { sorry };
      simp[H] at NOTLAST;
      contradiction;
    }
    case succ n' => {
      simp[NOTLAST];
      simp at NOTLAST;
      simp_arith at *;
      have H : v < Nat.succ n' := by {
          apply le_of_le_of_neq_upper_bound <;> simp_arith at * <;> simp <;> try assumption;
      };
      simp_arith at * <;> assumption;
    }
    }
}

-- keep the fin the same, just don't move it.
def Fin.lower (f: Fin (Nat.succ n)) (NOTLAST: f ≠ (Fin.last n)): Fin n :=
  { val := f.val, isLt := by {
      have H : _ := Fin.lt_n_of_not_last f NOTLAST;
      simp_arith[H];
    }
  }


-- lift of a lower is identity.
theorem Fin.lift_of_lower: ∀ (f: Fin (Nat.succ n)) (NEQ: f ≠ Fin.last _), lift (lower f NEQ) = f := by {
  intros f NEQ;
  simp[lift, lower];
}

-- lower of a lift is identity.
theorem Fin.lower_of_lift: ∀ (f: Fin (Nat.succ n))
  (NEQ: (lift f) ≠ Fin.last _ := Fin.lift_neq_last_succ f),
  Fin.lower f.lift NEQ = f := by {
  intros f NEQ;
  simp [lift, lower];
}
-- Get findom - last element
def Findom.init (f: Findom (Nat.succ n) α): Findom n α :=
  fun ix => f ix.lift

-- append to the end of a list.
def Findom.append (a: α) (f: Findom n α): Findom (Nat.succ n) α :=
  fun ix => if H:ix = (Fin.last n) then a else f (ix.lower H)

-- a findom is equal to its init appended with its lsat.
def Findom.eq_append_init_last: ∀ (f: Findom (Nat.succ n) α), f = f.init.append (f (Fin.last _)) := by {
  intros f;
  simp[Findom.append];
  congr;
  funext ix;
  simp;
  exact (match H:(decEq ix (Fin.last n)) with
        | isTrue HEQ => by {
          simp[HEQ];
        }
        | isFalse HNEQ => by {
          simp[HNEQ];
          simp[Findom.init];
          simp[Fin.lift_of_lower];
        });
}

-- the last value of an appended Findom is the element itself.
def Findom.last_of_append (a: α) (f: Findom n α): (f.append a) (Fin.last _) = a := by {
  simp[Findom.append];
}

-- this is an adjunction between lower and apend?
-- xs[lower i] = (xs ++ [k])[i] (recall that the lower is an artefact of types, the value does not change.)
def Findom.val_at_lower_equals_val_at_append (a: α) (f: Findom n α)
  (k: Fin (Nat.succ n)) (NEQ: k ≠ Fin.last _):
    f (Fin.lower k NEQ) = (f.append a) k  := by {
        simp;
        simp [Fin.lower, append];
        simp[NEQ];
}


-- the last value of the init of an appended Findom is the element itself.
def Findom.init_of_append (a: α) (f: Findom (Nat.succ n) α): (f.append a).init = f := by {
  simp[Findom.append, Findom.init];
  congr;
  funext ix;
  simp;
  have H : Fin.lift ix ≠ Fin.last _ := by apply Fin.lift_neq_last_succ;
  simp[H];
  simp;
  simp[Fin.lower_of_lift];
}

-- cast the domain of a findom along an equality.
def Findom.castDomain {n m: Nat} (f: Findom n α) (EQ: m = n): Findom m α :=
  EQ ▸ f

-- Convert a list into a finite domain function
def List.toFindom: (xs: List α) → Findom (xs.length) α
| xs => fun ix => xs.getF ix.val ix.isLt

-- Convert a finite domain function into a list.
def Findom.toList (fs: Findom n α): { xs: List α // xs.length = n }:=
  Subtype.mk ((List.rangeF n).map fs) (by {
     induction n;
     case zero => simp;
     case succ n' IH => simp;
  })

theorem toFindom_toList_id (xs: List α): xs.toFindom.toList = xs := by sorry
theorem toList_toFindom_id (fs: Findom n α):
  let xs  := fs.toList;  xs.property ▸ xs.val.toFindom = fs := by sorry

-- an endomorphism on Fin n
abbrev Endomorphism (n: Nat) := Findom n (Fin n)

def Endomorphism.id : Endomorphism n := fun ix => ix
-- g ∘ f = g after f
def Endomorphism.after (g f: Endomorphism n): Endomorphism n :=
  g∘ f

def Findom.toEndo (fs: Findom n (Fin n)): Endomorphism n :=
  fs

def List.mapFilter (f: a -> Option b): List a → List b
| [] => []
| x::xs => match f x with
           | .none => xs.mapFilter f
           | .some y => y :: xs.mapFilter f

@[simp]
def Mem.elimEmpty {y: α} (H: List.Mem y []): False := by {
  cases H;
}


#check Exists
def List.inMapFilter
  [DecidableEq b]  {f: a -> Option b} {xs: List a} {y: b}
  (Y: List.Mem y (xs.mapFilter f)) :
  ∃(x : a), List.Mem x xs ∧ (f x = .some y) := by {
  revert y;
  induction xs;
  case nil => {
    intros y Y;
    simp[mapFilter] at Y;
    have H : False := Mem.elimEmpty Y;
    contradiction;
  }
  case cons x' xs IH => {
    intros y Y;
    simp at *;
    simp [mapFilter] at Y;
    cases FX':(f x');
    case none => {
      simp [FX'] at Y;
      specialize (IH Y);
      cases IH;
      case intro val property => {
         constructor;
         constructor;
         apply Mem.tail;
         exact (property.left);
         exact property.right;
      }
    }
    case some y' => {
      simp[FX'] at Y;
      cases EQ:(decEq y y');
      case isFalse NEQ => {
        -- Since y /= y', we can't have y be member of (y' :: ..)
        cases Y;
        case head => {
          simp[NEQ];
          contradiction;
        }
        -- we must be in the tail, use the induction hypothesis.
        case tail Y => {
          specialize (IH Y);
          cases IH; -- @mathieu: how do I do this nicely?
          case intro x' X' => {
            apply (Exists.intro);
            constructor;
            apply Mem.tail;
            exact X'.left;
            exact X'.right;
          }

        }
      }
      case isTrue EQ' => {
        apply Exists.intro (w := x');
        constructor;
        apply Mem.head;
        simp[EQ'];
        exact FX';
      }

    }

  }

}

-- Characterize having '(List.Mem y (xs.mapFilter f)'  based on properties.
def List.inMapFilter'
  [DecidableEq b]  {f: a -> Option b} {xs: List a} {x : a} (MEM: List.Mem x xs) : ∀ {y: b}
  {FX: f x = .some y}, List.Mem y (xs.mapFilter f) := by {
  induction MEM;
  case head H x' xs' => {
       intros y FX;
       simp [mapFilter, FX];
       constructor;
  }
  case tail p q r s t u => {
    intros y FX;
    simp [mapFilter]
    cases FQ:(f q);
    case none => {
        simp;
        apply u;
        exact FX;
    }
    case some v => {
      simp;
      apply Mem.tail;
      apply u;
      exact FX;
    }
  }
}

-- Get the fibers of the endomorphism
-- TODO: get rid of this subtype.
def Endomorphism.fiber (e: Endomorphism n) (y: Fin n):
  List (Fin n) :=
  (List.rangeF n).mapFilter
     (fun i => if e i = y then .some i else .none)


-- Any element in the fiber does project down to `y`.
def Endomorphism.fiber_sound (e: Endomorphism n) (y: Fin n):
 ∀ x, x ∈ e.fiber y -> e x = y := by {
  intros x X;
  simp [fiber] at X;
  have H : _ := List.inMapFilter X;
  cases H;
  case intro x' X' => {
    have H : _ := And.right X';
    -- inMapFilter tells us that f(e(x')) = y must hold.
    cases (decEq (e x')  y);
    case isFalse NEQ => {
      simp [NEQ] at H;
    }
    case isTrue EQ => {
      simp [EQ] at H;
      rewrite [<- H];
      exact EQ;
    }
  }
}

-- every 'Fin n' is a member of 'List.rangeF n'
def List.mem_rangeF (x: Fin n): List.Mem x (List.rangeF n) := by {
 sorry -- TODO: prove this some rainy day.
}

-- the fiber of y contains all elements x such that f x = y.
-- This will allow us to prove uniqueness of toSection.
def Endomorphism.fiber_complete (e: Endomorphism n) (y: Fin n):
 ∀ x,  e x = y -> x ∈ e.fiber y := by {
  intros x X;
  simp[fiber];
  apply List.inMapFilter' (x := x);
  apply List.mem_rangeF;
  simp[X];
}





/-
def Findom.sequenceOptional {n: Nat} (fs: Findom n (Option α)): Option (Findom n α) :=
  match H:n with
  | 0 => .some { f := fun ix => ix.elim0 }
  | n'+1 => match fs.f 0 with
          | .some x =>
            match Findom.sequenceOptional (n := n') { f := fun ix => fs.f ix.lift} with
            | .some xs => .some { f := fun ix => match ix with
                                                | 0 => x
                                                | ix' => xs.f ix'
                                }
            | .none => .none
          | .none => .none
-/

def List.sequenceOptional: List (Option α) ->  Option (List α)
| [] => .some []
| x? :: xs => match x? with
      | .some x => match List.sequenceOptional xs with
                   | .some xs => .some (x :: xs)
                   | .none => .none
      | .none => .none

/-
Characterise what elements are in (List.sequenceOptional xs)
-/
theorem List.sequenceOptional_Mem {xs?: List (Option α)} {xs: List α}
  (XS: xs?.sequenceOptional = .some xs) {x: α} (MEM: List.Mem x xs):
  ∃ x? : Option α, List.Mem x? xs? ∧ x? = .some x := by {
  revert XS;
  revert xs?;
  induction MEM;
  case head y ys' => {
    intros xs? XS;
    induction xs?;
    case nil => {
      simp[sequenceOptional] at XS;
    }
    case cons x' xs' IH => {
      cases x';
      case none => {
        simp[sequenceOptional] at XS;
      }
      case some x'val => {
        simp[sequenceOptional] at XS;
        cases CONTRA:(sequenceOptional xs');
        case none => {
          simp[CONTRA] at XS;
        }
        case some xs'val => {
          simp[CONTRA] at XS;
          simp[XS];
          apply Exists.intro (w := .some x'val);
          constructor;
          have X'VAL: x'val = y := XS.left;
          simp[X'VAL];
          constructor;
          have X'VAL: x'val = y := XS.left;
          simp[X'VAL];
        }
      }
    }
  }
  case tail kk head' tail' MEMHEAD' MEMIH => {
    intros xs? XS;
    induction xs?;
    case nil => {
      simp[sequenceOptional] at XS;
    }
    case cons head'' tail'' IH2 => {
      cases HEAD'':head'';
      case none => {
        simp[HEAD''] at XS;
        simp[sequenceOptional] at XS;
      }
      case some head''val => {
        simp[HEAD''] at XS;
        simp[sequenceOptional] at XS;
        simp[IH2] at XS;
        cases CONTRA:(sequenceOptional tail'');
        case none => {
          simp[CONTRA] at XS;
        }
        case some xs'val => {
            simp[CONTRA] at XS;
            have XSRIGHT : xs'val = tail' := XS.right;
            have XSLEFT : head''val = kk := XS.left;
            simp[XSLEFT, XSRIGHT] at *;
            specialize (MEMIH CONTRA);
            cases MEMIH;
            case intro val property => {
              apply Exists.intro (w := val);
              constructor;
              apply Mem.tail;
              exact property.left;
              exact property.right;
            }
          }
        }
      }
    }
  }

/-
If sequenceOptional returns a value, then it has the same length.
-/
theorem List.sequenceOptional_length:
  ∀ (xs: List (Option α)) (xs': List α) (XS: xs.sequenceOptional = some xs'),
    length xs' = length xs := by {
   intros xs;
   induction xs;
   case nil =>  {
   intros  xs' XS;
   simp[sequenceOptional] at XS;
   simp[XS];
   rewrite [<- XS];
   simp;
   }
   case cons head tail IH => {
    intros  xs' XS;
    cases head;
    case none => {
      simp[sequenceOptional] at XS;
    }
    case some head' => {
      simp[sequenceOptional] at XS;
      cases REC:(sequenceOptional tail);
      case none => {
        simp[REC] at XS;
      }
      case some sequenceOptional => {
          simp[REC] at XS;
          simp[XS];
          rewrite[<- XS];
          simp;
          apply IH;
          exact REC;
      }
    }
   }
}

/-
-- traversable. What I really need is traverse.
-- Why does over half of this API reduce to having clever variants
-- of 'traverse'?
def Findom.everywhere_defined?
  (fs: Findom n (Option α)): Option (Findom n α) :=
  let xs := fs.toList
  have H : List.length xs.val = n:= xs.property;
  let xs'? := xs.val.sequenceOptional
  match XS':xs'? with -- how to rewrite?
  | .none => .none
  | .some xs' => have LEN: xs'.length = n := by {
       rewrite [<- H];
       apply List.sequenceOptional_length;
       simp [ XS'] at *;
      };  LEN ▸ xs'.toFindom

theorem Findom.everywhere_defined?_eval {fs?: Findom n (Option α)}
  {fs: Findom n α} (FINDOM: fs?.everywhere_defined? = .some fs) (i: Fin n):
    fs?.f i = .some (fs.f i) := by {
  simp[everywhere_defined?] at FINDOM;
  cases XS':List.sequenceOptional (toList fs?).val;
  case none => {
    simp[XS'] at FINDOM;
    sorry
  }
  case some fs => {
    simp[XS'] at FINDOM;
    sorry
  }
}
-/

def Findom.sequenceOptional {n: Nat} (fs: Findom n (Option α)): Option (Findom n α) :=
  match n with
  | 0 => .some Findom.nil
  | n+1 => match fs (Fin.last n) with
           | .none => .none
           | .some x => match Findom.sequenceOptional (Findom.init fs) with
                        | .none => .none
                        | .some fs' => .some (Findom.append x fs')


theorem Findom.sequenceOptional_append
  (fs?: Findom n (Option α))
  (fs: Findom n α)
  (FS: Findom.sequenceOptional fs? = fs)
  (a: α):
  Findom.sequenceOptional (fs?.append (.some a)) = .some (append a fs)  := by {
    simp[sequenceOptional];
    simp[Findom.last_of_append];
    cases n;
    case zero => {
        have FS?VAL : fs? = Findom.nil := by apply Findom.nil_unique;
        have FSVAL : fs = Findom.nil := by apply Findom.nil_unique;
        simp[FSVAL];
        simp [FS?VAL];
        simp[Findom.append];
        simp [Findom.init];
        simp[sequenceOptional];
    }
    case succ n' => {
      simp[Findom.init_of_append (a := some a)];
      simp[FS];
    }
  }



#print Nat.rec
#print List.rec

-- induction principle for findom.
theorem Findom.induction {α: Type} (motive: ∀ (n: Nat), Findom n α -> Prop):
  (NIL: motive 0 Findom.nil) -> (CONS: ∀ (x: α) (n: Nat) (f: Findom n α),
      motive n f -> motive (Nat.succ n) (Findom.append x f)) -> (f: Findom n α) -> motive n f := by {
  intros nil append;
  induction n;
  case zero => {
    simp[Findom, Findom.nil] at *;
    intros f;
    have H: f = Findom.nil := by {
      simp[Findom.nil_unique];
    };
    rewrite[H];
    apply nil;
  }
  case succ n' IH => {
    simp[Findom];
    intros f';
    have H : _ := Findom.eq_append_init_last (f := f')
    rewrite [H];
    apply append;
    apply IH;
  }
}

-- induction principle for findom where domain is always succ.
theorem Findom.inductionSucc {α: Type} (motive: ∀ (n: Nat), Findom (Nat.succ n) α -> Prop)
  (ONE: ∀ f: Findom (Nat.succ 0) α, motive _ f)  (IND: ∀ (x: α) (n: Nat) (f: Findom (Nat.succ n) α),
      motive n f -> motive _ (Findom.append x f)): ∀ (f: Findom (Nat.succ n) α), motive n f := by {
  induction n;
  case zero => {
    simp[Findom, Findom.nil] at *;
    intros f;
    apply ONE;
  }
  case succ n' IH => {
    simp[Findom];
    intros f';
    have H : _ := Findom.eq_append_init_last (f := f')
    rewrite [H];
    apply IND;
    apply IH;
  }
}


theorem Findom.sequenceOptional_is_some_everwhere {fs?: Findom n (Option α)}
  {fs: Findom n α} (FS: fs?.sequenceOptional = .some fs) (i: Fin n):
    fs? i = .some (fs i) := by {
      induction n;
      case zero => {
        simp[sequenceOptional] at FS;
        simp[FS];
        apply i.elim0;
      }
      case succ n' IH => {
        apply Findom.inductionSucc (motive :=
          fun (k: Nat) (gs?: Findom (Nat.succ k) (Option α)) => -- motive arguments
              ∀ (gs: Findom (Nat.succ k) α) -- impredicativity ftw
                (GS: sequenceOptional gs? = .some gs)
                (i: Fin (Nat.succ k)), -- our prop
                  gs? i = some (gs i));
        case ONE => {
          intros gs? gs GS k;
          simp [sequenceOptional] at GS;
          simp [Fin.last] at GS;
          cases EVAL:(gs? 0);
          case none =>  { simp [EVAL] at GS <;> contradiction };
          case some gs?0 => {
            simp[EVAL] at GS;
            simp [append] at GS;
            simp [GS];
            have KVAL : k = 0 := by apply Fin.one_unique;
            simp[KVAL] at *;
            have GSFN: gs = fun ix => gs?0 := by {
                simp at *;
                funext ix;
                rewrite [<- GS];
                simp;
                have IX: ix = 0 := Fin.one_unique _;
                simp[IX];
            }
            rewrite[GSFN] <;> simp[EVAL]
          }
        }
        case IND => {
          intros last?;
          intros size;
          intros gs? IH gs GS k;
          simp [append];
          cases (decEq k (Fin.last (Nat.succ size)));
          case isTrue KLAST => {
            simp[KLAST];
            simp [sequenceOptional] at GS;
            simp[Findom.last_of_append] at GS;
            cases LAST:last? <;> simp[LAST] at GS;
            case some last => {
              simp[LAST] at GS;
              simp[GS];
              simp[Findom.init_of_append] at GS;
              cases GSLAST:(gs? (Fin.last size)) <;> simp [GSLAST] at GS;
              case some gslast => {
                simp [GSLAST] at GS;
                cases RECINIT:(sequenceOptional (init gs?)) <;> simp[RECINIT] at GS;
                case some recval => {
                  rewrite[<- GS];
                  simp[Findom.last_of_append];
                }
              }
            }
          }
          case isFalse KLAST => {
            simp[KLAST];

            simp [sequenceOptional] at GS;
            simp[Findom.last_of_append] at GS;
            cases LAST:last?;
            case none => {
              simp[LAST] at GS;
            }
            case some last => {
              simp[LAST] at GS;
              simp[GS];
              simp[Findom.last_of_append] at GS;
              simp[Findom.init_of_append] at GS;
              cases GSLAST:(gs? (Fin.last size)) <;> simp[GSLAST] at GS;
              case some gslast => {
                cases SEQUENCE_INIT_GS:(sequenceOptional (init gs?)) <;> simp [SEQUENCE_INIT_GS] at GS;
                case some sequence_init_gs => {
                  -- we need to massage to apply IH, such that the RHS
                  -- can have a weird looking `gs`, as long as the LHS has a normal looking `gs?`.
                  -- TODO: consider making it the other way round?
                  rewrite [<- GS];
                  rewrite [<- Findom.val_at_lower_equals_val_at_append
                      (f := append gslast sequence_init_gs)
                      (NEQ := KLAST)];
                  apply IH;
                  rewrite [Findom.eq_append_init_last (f := gs?)];
                  rewrite[GSLAST];
                  apply Findom.sequenceOptional_append <;> assumption;
                }
              }
            }
          }
        }
        case GS => {
          apply FS;
        }
    }
}


def Findom.sequenceM {n: Nat} [M: Monad m] [LawfulMonad m]
  (fs: Findom n (m α)): m (Findom n α) :=
  match n with
  | 0 => pure Findom.nil
  | n+1 => do
          let x <- fs (Fin.last n)
          let fs' <- Findom.sequenceM (Findom.init fs)
          pure (Findom.append x fs')


theorem Findom.sequenceM_append [M: Monad m] [LAWFUL: LawfulMonad m]
  (fs?: Findom n (m α))
  (fs: Findom n α)
  (FS: Findom.sequenceM fs? = pure fs)
  (a: α):
  Findom.sequenceM (fs?.append (pure a)) = pure (append a fs)  := by {
    revert α;
    induction n;
    case zero => {
      intros α fs? fs FS a;
      simp[sequenceOptional] at FS;
      simp[FS];
      simp[sequenceM];
      have FSNIL: fs? = Findom.nil := Findom.nil_unique _;
      simp[FSNIL];
      simp[append];
      congr;
      funext x;
      have X: x = Fin.last 0 := Fin.one_unique _;
      simp[X];
    }
    case succ n' IH => {
      intros α fs? fs FS a;
      apply Findom.inductionSucc (motive :=
        fun (n: Nat) (fs?: Findom (Nat.succ n) (m α)) =>
          ∀(fs: Findom (Nat.succ n) α) (FS?: Findom.sequenceM fs? = pure fs) (a: α),
            Findom.sequenceM (fs?.append (pure a)) = pure (append a fs));
      case ONE => {
          intros gs? gs GS val;
          simp[sequenceM];
          simp[Findom.last_of_append];
          simp[Findom.init_of_append];
          simp[LAWFUL.seq_pure];
          simp[sequenceM] at GS;
          simp[sequenceM] at IH;
          simp[Findom.last_of_append] at IH;
          simp[Findom.init_of_append] at IH;
          sorry
      }
      case IND  => {
        intros mx n f IH fs K;
        intros a;
        simp[sequenceM];
        simp[Findom.init_of_append (a := some a)];
        simp[Findom.init_of_append];
        simp[Findom.last_of_append];
        simp[Findom.init_of_append];
        simp[FS, K];
        sorry
      }
      case FS? => { sorry }
    }
  }

-- This is kind of a misnomer, it rather returns the *unique section*
-- if such an object exists.
def Endomorphism.toSection? (e: Endomorphism n): Findom n (Option (Fin n)) :=
  fun y =>
    match e.fiber y with
    | [x] => .some x
    | _ => .none

@[simp]
theorem List.mem_singleton (a a': α) (H: List.Mem a  [a']): a = a' := by {
  cases H <;> simp at *;
  case tail IH => {
    simp[IH];
    have H : False := Mem.elimEmpty IH;
    contradiction;
  }

}

-- if we have a point where `s(i) = .some si`, then `si` is the unique
-- value in the codomain such that `pi(si) = i`
def Endomorphism.toSection?_is_unique (pi: Endomorphism n)
  (s: Findom n (Option (Fin n)))
  (S: pi.toSection? = s)
  (i: Fin n)
  (si: Fin n)
  (SI: s i = .some si)
  (sj: Fin n)
  (SJ: pi sj = i): sj = si := by {
   simp [Endomorphism.toSection?] at S;
   rewrite[<- S] at SI;
   simp at SI;
   cases FIBER:(Endomorphism.fiber pi i);
   case nil => { -- empty fiber, contradiction.
     simp [FIBER] at SI;
   }
   case cons head tail => {
     cases tail;
     case nil => { -- exactly 1 element in fiber, good case!
      simp [FIBER] at SI;
      rewrite[<- SI]; -- see that value of si is the unique value of the fiber.
      have IN: sj ∈ pi.fiber i := by {
           apply Endomorphism.fiber_complete;
           exact SJ;
      }
      simp[FIBER] at IN;
      simp at IN;
      apply List.mem_singleton; simp;
      exact IN;

    }
     case cons head' tail => { -- > 2 elements in fiber, contradiction.
       simp [FIBER] at SI;
     }
   }
}



-- Prove that if we manage to compute a section, then at every point
-- `i` that the section is defined, `pi(s(i)) = i`, upto `Option`
-- juggling.
-- The `option` juggling is necessary since we try to be nice.
def Endormophism.toSection?_is_section (pi: Endomorphism n)
  (s: Findom n (Option (Fin n)))
  (S: pi.toSection? = s)
  (i: Fin n)
  (si: Fin n)
  (SI: s i = .some si):
  pi si = i := by {
   simp [Endomorphism.toSection?] at S;
   rewrite[<- S] at SI;
   simp at SI;
   cases FIBER:(Endomorphism.fiber pi i);
   case nil => { -- empty fiber, contradiction.
     simp [FIBER] at SI;
   }
   case cons head tail => {
     cases tail;
     case nil => { -- exactly 1 element in fiber, good case!
      simp [FIBER] at SI;
      rewrite[<- SI]; -- see that value of si is the unique value of the fiber.
      apply Endomorphism.fiber_sound;
      rewrite [FIBER];
      constructor;

    }
     case cons head' tail => { -- > 2 elements in fiber, contradiction.
       simp [FIBER] at SI;
     }

   }
}

/-
TODO: consider using the galois connection given by defining
for a number 'x', the largest location 'y' such that (f y = x).

`(f y < x) v/s y < lloc x`
-/
/-
Dependently typed programming is like the expression problem.
We can either write Ohad/OOP, where we have data and proofs (behaviour)
next to each other. Or we can write in Xavier/functional style, where
the data is separate from the proofs (behaviour).
-/
def Endomorphism.inverse? (e: Endomorphism n): Option (Endomorphism n) :=
 let fs?: Findom n (Option (Fin n)) := e.toSection?
 let fs? := Findom.sequenceOptional fs?
 match fs? with
 | .none => .none
 | .some fs' => fs'.toEndo


theorem Endomorphism.inverse?_fg (f g: Endomorphism n)
  (G: f.inverse? = .some g): f.after g = id := by {
  simp[inverse?] at G;
  cases GLOBAL_SECTION:(Findom.sequenceOptional (toSection? f));
  case none => {
    simp[GLOBAL_SECTION] at G; -- contradiction
  }
  case some s => { -- global section
    simp[GLOBAL_SECTION] at G;

    sorry
  }
}

theorem Endomorphism.inverse?_gf (f g: Endomorphism n)
  (G: f.inverse? = .some g):  g.after f = id := by { sorry}





-- Convert a list into an endomorphism
def List.toEndo (xs: List (Fin n)) (LEN: n = length xs): Endomorphism n :=
  let fs : Findom n (Fin n) := xs.toFindom.castDomain LEN;
  fs.toEndo


-- Witnesses that the endomorphism 'f' is a permutation
structure Permutation (n: Nat) where
   f: Endomorphism n
   g: Endomorphism n
   gf: g ∘ f = id
   fg: f ∘ g = id

-- identity permutation
def Permutation.identity: Permutation n :=
 { f := id, g := id, gf := by { funext x; simp }, fg := by { funext x; simp } }


/-
#### 1D Tensors
-/

abbrev TensorIndex1D (n: Nat) := Fin n

structure Tensor1D where
  size0: Nat
  data: Findom size0 Int --  -> Int

def Tensor1D.isEq (v1 v2: Tensor1D): Decidable (v1 = v2) :=
  match decEq (v1.size0) (v2.size0) with
  | Decidable.isFalse prf =>
      Decidable.isFalse (by {
        intro H;
        cases H;
        contradiction;
      })
  | Decidable.isTrue SIZE0 =>
      Decidable.isTrue sorry


/-
### Primops that manipulate tensors.

These primitive operations are *constructive*, in that they build
simple tensors from other tensors by either manipulating the shape XOR the data,
never both. Decomposing other tensor transforms into these primitives
allows us to decompose the shape reasoning from the data reasoning.

All other operations must be written in terms of these primitives.
-/
def Tensor1D.empty: Tensor1D :=
{ size0 := 0, data := fun ix => ix.elim0 }

def Tensor1D.fill (t: Tensor1D) (cst: Int): Tensor1D :=  {
  size0 := t.size0
  data := fun _ => cst
}

-- Extract upto len `len` from the tensor.
def Tensor1D.extract (t: Tensor1D) (len: Nat) (LEN: len <= t.size0): Tensor1D :=
 {
    data := fun ix => t.data ix
 }


def Fin.offset (x: Fin n) (o: Nat): Fin (n + o) :=
  Fin.mk (x + o) (by { simp_arith; apply x.isLt; })

def TensorIndex1D.offset (ix: TensorIndex1D n) (o: Nat): TensorIndex1D (n + o) :=
    Fin.offset ix o

-- Offset the indexes into the tensor by `+offset`.
def Tensor1D.offset (t: Tensor1D) (offset: Nat) (OFFSET: offset <= t.size0): Tensor1D := {
  size0 := t.size0 - offset
  data := fun ix =>
     have H : t.size0 - offset + offset = t.size0 := by {
      apply Nat.sub_add_cancel <;> assumption
     };
     t.data (H ▸ ix.offset offset)
}

-- Stride indexes into the tensor by `*stride*.
/-
def Tensor1D.strided (t: Tensor1D) (stride: Nat): Tensor1D := {
  size0 := t.size0
  data := fun n => t.data (n * stride)
}
-/


/-
TODO: Build a theory that shows how to modify the *index* to be equivalent to the operation
on the *tensor*.
-/


instance : Inhabited Tensor1D where
  default := Tensor1D.empty

instance : ToString Tensor1D where
  toString t := "Tensor1D"

structure TensorIndex2D (size0: Nat) (size1: Nat): Type where
  ix0: Nat
  ix1: Nat
  IX0: ix0 < size0
  IX1: ix1 < size1


def TensorIndex2D.toFin: TensorIndex2D size0 size1 -> Fin (size0 * size1) := fun ix => {
  val := ix.ix0 * size1 + ix.ix1
  isLt := by {
    have IX0: ix.ix0 < size0 := ix.IX0;
    have IX1: ix.ix1 < size1 := ix.IX1;
    simp_arith;
    sorry
  }
}

-- subst, contradiction, assumption, simp.
def TensorIndex2D.ofFin: Fin (size0 * size1) -> TensorIndex2D size0 size1 := fun ix => {
  ix0 := (ix.val) / size1
  ix1 := ix.val % size1
  IX0 := by {
    have H: ix.val < size0 * size1 := ix.isLt;
    rewrite[Nat.div_lt_iff_lt_mul] <;> simp[H];
    cases size1 <;> simp_arith at * <;> try contradiction;
  }
  IX1 := by {
      apply Nat.mod_lt;
      cases size1 <;> simp_arith at * <;> try contradiction;
      apply Fin.elim0 <;> assumption;
  }
}

def TensorIndex2D.toFin_ofFin_eq:
  ∀ (t: TensorIndex2D size0 size1), TensorIndex2D.ofFin t.toFin = t := by {
    intros t;
    simp [TensorIndex2D.toFin, TensorIndex2D.ofFin];
    cases t;
    case mk ix0' ix1' IX0' IX1' => {
      simp_arith;
      constructor;
      sorry
      sorry
    }

  }
/-
A subview into a 2D tensor.
-/
structure TensorSubview2D (maxsize0: Nat) (maxsize1: Nat): Type where
  -- ix0: Nat
  -- ix1: Nat
  size0: Nat
  size1: Nat
  IX0: size0 <= maxsize0
  IX1: size1 <= maxsize1





-- enlarge the tensor index to index a larger space.
def TensorIndex2D.enlarge {size0 size0' size1 size1': Nat}
  (INC0: size0 <= size0') (INC1: size1 <= size1')
  (ix: TensorIndex2D size0 size1): TensorIndex2D size0' size1' := {
    ix0 := ix.ix0
    ix1 := ix.ix1
    IX0 := by {
        have H: ix.ix0 < size0 := ix.IX0;
        simp_arith;
        apply Nat.lt_of_lt_of_le H INC0;
        }
    IX1 := by {
        have H: ix.ix1 < size1 := ix.IX1;
        simp_arith;
        apply Nat.lt_of_lt_of_le H INC1;
    }
  }
def TensorIndex2D.transpose
  (ix: TensorIndex2D size0 size1): TensorIndex2D size1 size0 := {
    ix0 := ix.ix1
    ix1 := ix.ix0
    IX0 := ix.IX1
    IX1 := ix.IX0
  }

lemma Nat.lt_mul_cancel_left (a b x: Nat) (H: a < b) (XNEQ0: 0 < x): a * x < b * x := by sorry_arith;

def TensorIndex2D.stride (ix: TensorIndex2D size0 size1) (stride0: Nat) (STRIDE0: 0 < stride0)
  (stride1: Nat) (STRIDE1: 0 < stride1): TensorIndex2D (size0*stride0) (size1*stride1) := {
  ix0 := ix.ix0 * stride0
  ix1 := ix.ix1 * stride1
  IX0 := by {
      have H: ix.ix0 < size0 := ix.IX0;
      apply Nat.lt_mul_cancel_left <;> assumption
     }
  IX1 := by {
      have H: ix.ix1 < size1 := ix.IX1;
      apply Nat.lt_mul_cancel_left <;> assumption
  }
}

/-
2D Tensors
-/
structure Tensor2D where
  size0: Nat
  size1: Nat
  /- Switch to using TensorIndex? -/
  data: TensorIndex2D size0 size1 -> Int

/-
def decideEqData (f g: TensorIndex2D size0 size1 -> Int): Decidable (f = g) :=
-/

#check DecidableEq
def Tensor2D.isEq (v1 v2: Tensor2D): Decidable (v1 = v2) :=
  match decEq (v1.size0) (v2.size0) with
  | Decidable.isTrue SIZE0 =>
      match decEq (v1.size1) (v2.size1) with
      | Decidable.isTrue SIZE1 => Decidable.isTrue sorry
      | Decidable.isFalse prf =>
          Decidable.isFalse (by {
            intro H;
            cases H;
            contradiction;
          })
  | Decidable.isFalse prf =>
      Decidable.isFalse (by {
        intro H;
        cases H;
        contradiction;
      })

def Tensor2D.empty: Tensor2D :=
  { size0 := 0, size1 := 0, data := fun ix => by {
      have CONTRA: ix.ix0 < 0 := ix.IX0;
      simp[Nat.not_lt_zero] at CONTRA;
    }
  }


/-
from a subview of size nxm, extract out a tensor of size nxm,
given a larger tensor of size (t.size0 x t.size1)
-/
def TensorSubview2D.extract (view: TensorSubview2D n m)
  (t: Tensor2D)
  (HN: n <= t.size0) (HM: m <= t.size1): Tensor2D  :=
  Tensor2D.mk view.size0 view.size1
    (fun ix => t.data (ix.enlarge
        (by {
          have HMID : view.size0 <= n := view.IX0;
          apply Nat.le_trans;
          apply HMID;
          apply HN;
        }) (by {
          have HMID : view.size1 <= m := view.IX1;
          apply Nat.le_trans;
          apply HMID;
          apply HM;
        })))

def Tensor2D.extractSubview (t: Tensor2D) (subview: TensorSubview2D t.size0 t.size1):
  Tensor2D  := Tensor2D.mk subview.size0 subview.size1
    (fun ix => (subview.extract t (by simp) (by simp)).data ix )

instance : Inhabited Tensor2D where
  default := Tensor2D.empty

instance : ToString Tensor2D where
  toString t := "Tensor2D"


/-
Create a tensor2d filled with the same value.
-/
def Tensor2D.fill (t: Tensor2D) (val: Int): Tensor2D :=
  Tensor2D.mk t.size0 t.size1 (fun _ix => val)

def Tensor2D.extractslice
  (t: Tensor2D)
  (size0 size1: Nat)
  (SIZE0: size0 <= t.size0) (SIZE1: size1 <= t.size1): Tensor2D :=
   Tensor2D.mk size0 size1
    (fun ix => t.data (ix.enlarge SIZE0 SIZE1))


def Tensor2D.extractslice' (large: Tensor2D)
  (subview: TensorSubview2D large.size0 large.size1): Tensor2D :=
  Tensor2D.mk subview.size0 subview.size1
    (fun ix => large.data (ix.enlarge subview.IX0 subview.IX1))

-- Transpose of a tensor by swapping indexes
def Tensor2D.transpose (t: Tensor2D): Tensor2D :=
  Tensor2D.mk t.size1 t.size0 (fun ix => t.data ix.transpose)

-- Stride index into a tensor, scaling the indexing by `stride0, stride1`.
def Tensor2D.stride (t: Tensor2D) (stride0 stride1: Nat)
  (STRIDE0: 0 < stride0) (STRIDE1:  0 < stride1): Tensor2D :=
  Tensor2D.mk (t.size0 / stride0) (t.size1 / stride1)
    (fun ix => t.data <| (ix.stride stride0 STRIDE0 stride1 STRIDE1).enlarge
    (by { rewrite[<- Nat.le_div_iff_mul_le]; simp_arith; apply STRIDE0; })
    (by { rewrite[<- Nat.le_div_iff_mul_le]; simp_arith; apply STRIDE1; }))


def Tensor2D.toSubview (t: Tensor2D): TensorSubview2D t.size0 t.size1 :=  {
    size0 := t.size0,
    size1 := t.size1,
    IX0 := by simp,
    IX1 := by simp,
  }


def TensorIndex2D.isInSubview (t: Tensor2D) (subview: TensorSubview2D t.size0 t.size1)
  (ix: TensorIndex2D t.size0 t.size1):
  Option (TensorIndex2D subview.size0 subview.size1) :=
  dite (ix.ix0 < subview.size0)
  (fun LT0 =>
    dite (ix.ix1 < subview.size1)
    (fun LT1 =>
      .some (TensorIndex2D.mk ix.ix0 ix.ix1 LT0 LT1)
    )
    (fun GEQ1 => .none))
  (fun GEQ0 => .none)

/-
Represents that `small` is located inside a slice of `large`.
-/
structure TensorSlice2D (small large: Tensor2D) where
  SIZE0: small.size0 <= large.size0
  SIZE1: small.size1 <= large.size1

def TensorSlice2D.toSubview (slice: TensorSlice2D small large):
  TensorSubview2D large.size0 large.size1 := {
    size0 := small.size0,
    size1 := small.size1,
    IX0 := slice.SIZE0,
    IX1 := slice.SIZE1 }

/-
4D Tensors
-/
structure Tensor4D where
  size0: Nat
  size1: Nat
  shape2: Nat
  shape3: Nat
  data: List Int -- monomorphic tensors
  h_data_size: data.length = (size0 * size1 * shape2 * shape3)


def Tensor4D.isEq (v1 v2: Tensor4D): Decidable (v1 = v2) := by {
  cases v1;
  cases v2;
  simp;
  exact inferInstance;
}

def Tensor4D.empty: Tensor4D :=
  { size0 := 0, size1 := 0, shape2 := 0, shape3 := 0, data := [], h_data_size := rfl }

instance : Inhabited Tensor4D where
  default := Tensor4D.empty

instance : ToString Tensor4D where
  toString t := "Tensor4D"


/-
### shapeProd
-/

def shapeProd: List Nat → Nat :=
  List.foldr (·*·) 1

theorem shape_prod_nil: shapeProd (0::l) = 0 := by
  induction l <;> simp [shapeProd, List.foldr]

@[simp]
theorem shapeProd.cons_unfold: ∀ (x: Nat) (xs: List Nat),
  shapeProd (x :: xs) = x * shapeProd xs := by {
   intros x xs;
   simp [shapeProd, List.foldr];
}

/-
### Flat tensor index
-/
/-
A 1D index into a tensor. Witnesses that the flat index is in bounds of the shape of the tensor.
-/
abbrev TensorFlatIndex (bound: Nat) := Fin bound

def TensorFlatIndex.eq_proof_irrelevant  (f1: TensorFlatIndex b) (f2: TensorFlatIndex b) (IXEQ: f1.val = f2.val): f1 = f2 := by {
  induction f1;
  case mk ix1 H1 => {
  induction f2;
  case mk ix2 H2 => {
   simp [IXEQ];
  }
  }
}


def TensorFlatIndex.cast_left: ∀ (bound bound': ℕ) (EQ: bound = bound') (ix: ℕ) (prf: ix < bound) (prf': ix < bound'),
  EQ ▸ { val := ix, isLt := prf : TensorFlatIndex bound } = {val := ix, isLt := prf' }
   := by {
  intros bound bound';
  intros EQ ix prf prf';
  cases EQ;
  simp;
}

def TensorFlatIndex.cast_right: ∀ (bound bound': ℕ) (EQ: bound = bound') (ix: ℕ) (prf: ix < bound) (prf': ix < bound'),
  { val := ix, isLt := prf : TensorFlatIndex bound } = EQ ▸ {val := ix, isLt := prf' }
   := by {
  intros bound bound';
  intros EQ ix prf prf';
  cases EQ;
  simp;
}

theorem TensorFlatIndex.bound_non_zero (flat: TensorFlatIndex bound): bound ≠ 0 := by {
  intros BOUND;
  have H_INBOUND := flat.isLt;
  simp [BOUND] at H_INBOUND;
  simp [Nat.not_lt_zero] at H_INBOUND;
}

theorem TensorFlatIndex.bound_zero_absurd (flat: TensorFlatIndex 0): False := by {
  have H_INBOUND := flat.isLt;
  simp [Nat.not_lt_zero] at H_INBOUND;
}

@[simp]
theorem Nat.succ_gt_zero (n: Nat): Nat.succ n > 0 := by {
  simp [GT.gt];
}

@[simp]
theorem Nat.nonzero_iff_gt_zero: ∀ (n: Nat), n ≠ 0 <-> n > 0 := by {
  intros n;
  constructor;
  case mp => {
  intros NEQ_0;
  cases n;
  case zero => {
    contradiction;
  }
  case succ n' => { simp [Nat.succ_gt_zero]; }
  }
  case mpr => {
   intros GT_ZERO;
   cases n;
   case zero => {
     simp at GT_ZERO;
   }
   case succ n' => { simp; }
  }
}

-- Bound is always greater than zero.
theorem TensorFlatIndex.bound_gt_zero(flat: TensorFlatIndex bound): bound > 0 := by {
  have BOUND_NONZERO: bound ≠ 0 := TensorFlatIndex.bound_non_zero flat;
  cases bound;
  case zero => {
    simp [Nat.zero, BOUND_NONZERO];
    contradiction;
  }
  case succ bound' => {
    apply Nat.succ_gt_zero;
  }
}

@[simp]
theorem Nat.mul_nonzero_implies_left_nonzero: ∀ (a b: Nat) (NEQ: a * b ≠ 0), a ≠ 0 := by {
  intros a b NEQ;
  induction a;
  case zero => {
   simp at NEQ;
  }
  case succ a' IH => {
    apply Nat.succ_ne_zero;
  }
}

@[simp]
theorem Nat.mul_nonzero_implies_right_nonzero: ∀ (a b : Nat) (NEQ: a * b ≠ 0), b ≠ 0 := by {
  intros a b NEQ;
  induction a;
  case zero => {
   simp at NEQ;
  }
  case succ a' IH => {
    induction b;
    case zero => {
     simp at NEQ;
    }
    case succ b' IH => {
     apply Nat.succ_ne_zero;
    }
  }
}

-- if product of number is nonzero, then every element is nonzero
theorem shapeProd_nonzero_implies_member_nonzero: ∀ (xs: List Nat)
   (x: Nat) (MEM: List.Mem x xs) (PROD: shapeProd xs > 0) , x > 0 := by {
   intros xs x MEM;
   induction MEM;
   case head a as => {
     simp [shapeProd, List.foldr];
     intros H;
     rewrite [<- Nat.nonzero_iff_gt_zero];
     apply Nat.mul_nonzero_implies_left_nonzero;
     rewrite [<- Nat.nonzero_iff_gt_zero] at H;
     apply H;
   }
   case tail b bs MEM IH => {
     intros H;
     apply IH;
     simp at H;
     rewrite [<- Nat.nonzero_iff_gt_zero] at *;
     apply (Nat.mul_nonzero_implies_right_nonzero);
     apply H;
   }
}


-- A TensorFlatIndex of a shapeProd will be nonzero.
theorem TensorFlatIndex.shapeProd_member_nonzero
  (shape: List Nat)
  (flat: TensorFlatIndex (shapeProd shape))
  (n: Nat) (MEMBER: List.Mem n shape): n > 0 := by {
  have PROD_NONZERO: shapeProd shape > 0 := flat.bound_gt_zero;
  apply shapeProd_nonzero_implies_member_nonzero;
  exact MEMBER;
  exact PROD_NONZERO;
}


@[simp]
theorem Nat.mod_zero_implies_div_mul_equal (n: Nat) (modulus: Nat)
  (MODZERO: n % modulus = 0): (n / modulus) * modulus = n := by {
  have MULTIPLE: n = 0 + (n / modulus) * modulus := by {
    rewrite [<- MODZERO];
    rewrite [Nat.mul_comm];
    simp [Nat.mod_add_div];
  }
  simp at MULTIPLE;
  rewrite [<- MULTIPLE];
  rfl;
}

@[simp]
theorem Nat.mul_cancel_right (n m: Nat) (MODZERO: n % m = 0): (n / m) * m = n := by {
    rewrite [Nat.mod_zero_implies_div_mul_equal n m MODZERO];
    rfl;
}

@[simp]
theorem Nat.div_lt_if_mod (ix bound modulus: Nat) (IX: ix < bound) (MODULUS: modulus > 0) (DIV: bound % modulus = 0):
  ix / modulus < bound / modulus := by {
  rewrite [Nat.div_lt_iff_lt_mul, Nat.mul_cancel_right];
  apply IX;
  apply DIV;
  apply MODULUS;
}

-- A theory of splitting and merging
-- 'TensorFlatIndex'es. This will be used to provide a theory
-- of delinearizing arbitrary tensor indexes
-- in terms of TensorFlatIndexes.
-- Split a TensorFlatIndex into two
def TensorFlatIndex.split
  (n modulus: Nat) (MODULUS: modulus > 0) (DIV: n % modulus = 0)
  (flat: TensorFlatIndex n): (TensorFlatIndex modulus) × (TensorFlatIndex (n/modulus)) :=
  (Fin.mk (flat.val %  modulus) (Nat.mod_lt flat.val MODULUS),
   Fin.mk (flat.val / modulus) (Nat.div_lt_if_mod flat.val n modulus flat.isLt MODULUS DIV))

theorem Nat.le_pred_if_lt (x n : Nat) (X_LT_N: x < n): x <= pred n := by {
     cases n;
     case zero => { simp [Nat.not_lt_zero] at X_LT_N; }
     case succ n' => {
      rewrite [Nat.pred_succ];
      apply Nat.le_of_lt_succ;
      exact X_LT_N;
    }
}

theorem Nat.le_one_minus_if_lt (x n : Nat) (X_LT_N: x < n): x <= pred n := by {
    rewrite [<- Nat.sub_one];
    apply Nat.le_pred_if_lt;
    simp; exact X_LT_N;
}

theorem Nat.le_mul_pred (x y n: Nat) (LE: x <= Nat.pred n): x * y <= n * y - y := by {
   cases H:n;
   case zero => {
   rewrite [H] at LE;
   simp at LE;
   rewrite [LE];
   simp;
   }
   case succ n' => {
   simp at LE;
   rewrite [H] at LE;
   simp at LE;
   sorry; -- algebra to be done.
   }
}

-- x < n <=> x <= n - 1
-- #check Nat.lt_of_succ_le
-- Merge a TensorFlatIndex into a large TensorFlatIndex
def TensorFlatIndex.merge
  (flat0: TensorFlatIndex N0)
  (flat1: TensorFlatIndex N1): TensorFlatIndex (N0 * N1) :=
  Fin.mk (flat1.val * N1 + flat0.val) (by {
     have IX0: flat0.val <= Nat.pred N0 := Nat.le_pred_if_lt _ _ flat0.isLt;
     have IX1: flat1.val <= Nat.pred N1 := Nat.le_pred_if_lt _ _ flat1.isLt;
     have IX0_N: flat0.val * N1 <= N0 * N1 - N1 := by {
      apply Nat.le_mul_pred <;> simp;
      exact IX0;
     }
     -- algebra
     sorry
  })

/-
Fully generic ND index. Currently unused.
-/
inductive TensorIndex': List Nat -> Type :=
|  Empty: TensorIndex' []
|  Dim (bound0: Nat)
      (ix: TensorFlatIndex bound0)
      (rest: TensorIndex' shape): TensorIndex' (bound0 :: shape)


/-
Projecting out outermost dimension
-/
def TensorIndex'.projectOut
  {outermost: Nat}
  {shape: List Nat}
  (index: TensorIndex' (outermost :: shape)): TensorIndex' shape :=
  match index with
  | .Dim _ _ rest => rest

inductive List.NonEmpty: List α-> Prop where
| Singleton (a: α): List.NonEmpty [a]
| Cons (a: α) (AS: List.NonEmpty as): List.NonEmpty (a::as)


theorem List.NonEmpty.empty_absurd (α: Type) (CONTRA: List.NonEmpty (@List.nil α)): False := by {
  cases CONTRA;
}

@[simp]
theorem TensorIndex'.empty_dims_is_empty (index: TensorIndex' []): index = .Empty := by {
  cases index; simp;
}

@[reducible, simp]
def TensorIndex'.getLinearizedIndexNumber
   {dims: List Nat} (index: TensorIndex' dims) : TensorFlatIndex (shapeProd dims) :=
    match index with
    | .Empty =>  Fin.mk 0 (by {simp[shapeProd];})
    | .Dim bound0 ix rest => ix.merge rest.getLinearizedIndexNumber


theorem Nat.lt_iff_gt: ∀ (a: Nat) (b: Nat), a < b <-> b > a := by {
  intros a b; constructor;
  case mp => {
     intros A_LT_B;
     simp [GT.gt]; exact A_LT_B;
  }
  case mpr => {
    intros B_GT_A;
    simp [GT.gt] at B_GT_A;
    exact B_GT_A;
  }
}



/-
### Naivete of definition of delineralizatoin

One might initially choose to believe that for ANY modulus, we can delin
def TensorIndex.delinearizeInnermost {innerDim: Nat} {restDims: List Nat}
  (modulus: Nat)
  (index: TensorIndex (innerDim :: restDims)):
    TensorIndex (modulus :: (innerDim/modulus) :: restDims) :=

This is absurd, because I Can choose modulus to be super large (9999), then
the tensor collapses because (innerDim/modulus) becomes = 0.


As a second try, one can try to add the assumtion that (modulus < innerDim).
This too is insufficient!
For the shape:
  (modulus, innerDim / modulus, ...)
we would naively choose the indexes:
  (innermostix % modulus, innermostix / modulus)

The 0th entry is clearly inbounds:
  (innermostix % modulus) < modulus

the 1st entry is not necessarily inbounds!
    innermostix / modulus < innerDim / modulus ??
   Even given that (innermostix < innerDim) from the original tensor, we cannot
   conclude that division preserves less than!
   eg 2 < 3 =/=> (2/9999) < (3/9999)!


We need some kind of divisibility criterion.
-/

theorem shapeProd_cons_prod (x y: Nat) (zs: List Nat): shapeProd (x :: y :: zs) = shapeProd ((x *y) :: zs) := by {
   simp [Nat.mul_assoc];
}


-- Build a 1D TensorIndex from a FlatIndex
def TensorIndex'.ofFlatIndex1D {innerDim: Nat}
  (flat: TensorFlatIndex innerDim): TensorIndex' [innerDim] := .Dim innerDim flat .Empty

theorem Nat.mul_of_nonzero_is_nonzero: ∀ (a b: Nat) (A: a ≠ 0) (B: b ≠ 0), a * b ≠ 0 := by {
   intros a;
   induction a;
   case zero => {
     intros b A_NEQ_ZERO; simp [A_NEQ_ZERO]; contradiction;
   }
   case succ a' IH => {
     intros b;
     induction b;
     case zero => {
        intros A B;
        simp at B;
     }
     case succ b' IH' => {
      intros A B;
      simp [Nat.mul];
      rewrite [Nat.nonzero_iff_gt_zero] at *;
      simp [Nat.mul_pos];
    }
   }

}


-- Helper function to zip a list with the index of the current value
def zipFlatIndexGo (xs: List α) (ix: Nat) (bound: Nat) (H: ix + xs.length = bound): List (α × TensorFlatIndex bound) :=
  match xs with
  | [] => []
  | x::xs' =>
     let ix_inbounds : ix < bound := by {
      rewrite [← H];
      apply Nat.lt_add_of_pos_right;
      simp;
     }
     let ix' := ix + 1
     let H' :ix' + xs'.length = bound := by {
       rewrite [← H];
       simp;
       rewrite [Nat.succ_eq_add_one];
       -- ⊢ ix + 1 + List.length xs' = ix + (List.length xs' + 1)
       have SWIZZLE : (1 + List.length xs' = List.length xs' + 1) := by simp[Nat.add_comm];
       rewrite [Nat.add_assoc];
       rewrite [SWIZZLE];
       simp;
     }
     (x, Fin.mk ix ix_inbounds) :: zipFlatIndexGo xs' ix' bound H'




-- zipFlatIndexGo maintains length of the list.
theorem zip_flat_index_go_length (xs: List α): ∀ (ix: Nat) (bound: Nat) (H: ix + xs.length = bound),
  xs.length = (zipFlatIndexGo xs ix bound H).length := by {
  induction xs;
  case nil => {
    intros; unfold zipFlatIndexGo; rfl;
  }
  case cons x xs' IND => {
    intros ix bound H;
    simp [zipFlatIndexGo];
    apply IND;
  }
}
#check Nat.zero_lt_of_lt


-- The value of the (zipFlatIndexGo xs ix bound ...):
--   ie, we have a list of total length `bound`, we have read list upto index `ix`, and the rest of the list is `xs`,
--   must be (ix + deltaIx).
theorem List.zip_flat_index_go_get: ∀ (xs: List α) (ix: Nat) (bound: Nat) (H: ix + xs.length = bound)
  (deltaIx: Nat) (GETIX: deltaIx < xs.length),
  ((zipFlatIndexGo xs ix bound H).getF deltaIx (zip_flat_index_go_length xs ix bound H ▸ GETIX)) =
  (xs.getF deltaIx GETIX, Fin.mk (n := bound)
                           (val := ix + deltaIx)
                           (isLt := by { rewrite [<- H]; simp [Nat.add_lt_add_left, GETIX]; } )) := by {
  intros xs;
  induction xs;
  case nil => {
      intros ix bound H deltaIx GETIX;
      simp [List.length, Nat.not_lt_zero] at GETIX;
  }
  case cons x xs' IND => {
   intros ix bound H deltaIx GETIX; -- consider pulling deltaIx earlier
   cases deltaIx;
   case zero => {
      simp;
      simp [zipFlatIndexGo, List.getF]
   }
   case succ deltaIx' => {
     simp [zipFlatIndexGo];
     simp [List.getF];
     rewrite [IND];
     simp [Nat.add_assoc, Nat.add_one, Nat.succ_add, Nat.add_succ];
     simp at GETIX;
     apply Nat.lt_of_succ_lt_succ;
     exact GETIX;
   }
  }
}

-- Zip a list with the index of the current value
def List.zipFlatIndex (xs: List α): List (α × TensorFlatIndex xs.length) :=
  zipFlatIndexGo xs 0 (H := by simp)


-- zipFlatIndex preserves length of the list
@[simp]
theorem List.length_zip_flat_index (xs: List α): length (List.zipFlatIndex xs) = length xs := by {
  apply Eq.symm;
  apply zip_flat_index_go_length;
}

-- The correctness of `List.zipFlatIndex`: value that it zips is the index of the element.
theorem List.zip_flat_index_get (xs: List α) (getIx: Nat) (GETIX: getIx < xs.length):
  (List.getF (List.zipFlatIndex xs) getIx (by simp; apply GETIX)) = (List.getF xs getIx GETIX, Fin.mk (n := xs.length) getIx GETIX) := by {
  simp[zipFlatIndex];
  have RHS :  { val := getIx, isLt := GETIX : TensorFlatIndex (xs.length) } = {val := 0 + getIx, isLt := by { simp; apply GETIX } : TensorFlatIndex (xs.length)} := by {
    simp;
  }
  rewrite [RHS];
  apply List.zip_flat_index_go_get (xs := xs) (ix := 0) (bound := List.length xs) (deltaIx := getIx) (GETIX := GETIX);
}
/-
3D Tensors
-/

structure TensorIndex3D (sizes: Fin 3 -> Nat): Type where
  dim2ix: (dim: Fin 3) -> Fin (sizes dim) -- given [0,1,2], return the index value

structure Tensor3D where
  sizes: Fin 3 → Nat
  data: TensorIndex3D sizes -> Int


-- permute the tensor index by `f`.
def TensorIndex3D.permute
   (f: (Fin 3) → (Fin 3)): TensorIndex3D sizes -> TensorIndex3D (sizes ∘ f)
| TensorIndex3D.mk dim2ix => TensorIndex3D.mk (fun dim => dim2ix (f dim))


theorem comp_assoc: f ∘ (g ∘ h) = (f ∘ g) ∘ h := by {
  funext x;
  simp[Function.comp];
}

theorem Permutation.simp_left (P: Permutation n): (k ∘ P.f) ∘ P.g = k := by {
   rewrite[<- comp_assoc];
   simp[P.fg];
   funext x;
   simp;
}


-- Permute the tensor dimensions  by f.
def Tensor3D.permute (P: Permutation 3): Tensor3D -> Tensor3D
| Tensor3D.mk sizes data =>
  Tensor3D.mk (sizes ∘ P.f) (fun ix => by {
    let ix' := ix.permute P.g
    have H : (sizes ∘ P.f) ∘ P.g = sizes := P.simp_left
    have ix'' : TensorIndex3D sizes := H ▸ ix'
    exact (data ix'')
   })

#print Tensor3D.permute

/-
Potential ways to represent permutations:
- function (Fin n -> Fin n) with given inverse
- a List of naturals of length n with no repeats.

How to check if a list is a permutation:
Check if each number in [0..n-1] occurs exactly once in the list.

How to check if a function is a permutation:
-
-/
-- Create a nat to a Fin value, if it is within the Fin.
def Nat.toFin (lim: Nat) (n: Nat): Option (Fin lim) :=
  if H: n < lim
  then .some (⟨n, H⟩)
  else .none

-- Convert a list of Nat to a list of Fin
def ListNat2ListFin (lim: Nat): List Nat -> Option (List (Fin lim))
| [] => .some []
| x::xs =>
      match Nat.toFin lim x with
      | .none => .none
      | .some y =>  match ListNat2ListFin lim xs with
                | .none => .none
                | .some ys' => y::ys'

-- Loop to get the index of an element in the function
def finGetIndexOf_loop [DecidableEq a] (i: Nat) (I: i < n)
  (f: Fin n -> a) (x: a): Option (Fin n)  :=
  if f ⟨i, I⟩ == x then .some ⟨i, I⟩ else
  match H:i with
  | 0 => .none
  | i'+1 => finGetIndexOf_loop i'
         (by { simp_arith; apply le_of_lt; exact I; })
         f x

-- Get the index of an element in the function.
def finGetIndexOf [DecidableEq a] (f: Fin n -> a) (x: a): Option (Fin n) :=
  match H:n with
  | 0 => .none
  | n'+1 => finGetIndexOf_loop n' (by { simp_arith; }) f x



-- Create a function (Fin n -> Option (Fin n))
-- def mkInversePointwise?: (Fin n -> Fin n) -> Fin n -> Option (Fin n)
-- | f, ix => finGetIndexOf f ix
--
-- def mkInverseGlobal?: (Fin n -> Fin n) -> Option (Fin n -> Fin n)
-- | f, ix => FinOption2OptionFin (mkInversePointwise? f ix)


/-
Tensor1d.mapM
-/

def List.mapMLengthProof {M: Type -> Type} {α β: Type} [Mon: Monad M] [LawfulMonad M]
  (f: α → M β) (xs: List α):
  M { ys :  (List β) // (List.length ys = xs.length) } :=
  match xs with
  | [] => pure ⟨[], by { simp }⟩
  | x::xs' => do
      let ys' <- List.mapMLengthProof f xs'
      let y <- f x
      return ⟨y::ys'.val, by {
        simp[List.length];
        have H: length ys'.val = length xs' := ys'.property
        exact H;
    }⟩


def Findom.map (f: α → β) (l: Findom n α): Findom n β := f ∘ l

-- mapM for Findom.
def Findom.mapM {M: Type → Type} {α: Type} [Monad M] [LawfulMonad M]
  (findom: Findom n α) (f: α → M β):
  M (Findom n β) := do
  let l := findom.toList
  have H : List.length l.val = n := l.property
  let out : { ys // List.length ys = List.length l.val } <- List.mapMLengthProof f l.val
  have H' : List.length out.val = List.length l.val := out.property
  have H'': List.length out.val = n := Eq.trans H' H
  return (H'' ▸ List.toFindom out.val)


-- theorem List.mapLength (f: a -> b) (xs: List a): List.length (List.map f xs) = List.length xs := by {
--   induction xs;
--   case nil => simp[List.length]
--   case cons x xs' IH => simp[List.length];
-- }

def Findom.mapMWithIndex {M: Type → Type} {α: Type} [Monad M] [LawfulMonad M]
  (findom: Findom n α) (f: α × Fin n → M β):
  M (Findom n β) := do
  let l := findom.toList
  have L_LENGTH_EQ_N : l.val.length = n := l.property

  let ys : List (α × TensorFlatIndex (List.length l.val)) := List.zipFlatIndex l.val
  let ys' : List (α × TensorFlatIndex n) := ys.map (L_LENGTH_EQ_N ▸ .)
  let YS'_LENGTH_EQ_N : ys'.length = n := sorry
  let out : { out // List.length out = List.length ys' } <- List.mapMLengthProof f ys'
  have OUT_LENGTH_EQ_N: List.length out.val = n := Eq.trans out.property YS'_LENGTH_EQ_N
  return (OUT_LENGTH_EQ_N ▸ List.toFindom out.val)


theorem Findom.mapM_map [Monad M] [LawfulMonad M] (l: Findom n α) (f: α → β) (fM: α → M β)
      (F: forall a, fM a = pure (f a)):
    l.mapM fM = return l.map f := by {
      apply Findom.induction (motive := fun n l => l.mapM fM = return l.map f);
      case NIL => {
        simp[Findom.mapM, Findom.map];
        simp[Findom.toList];
        simp[List.mapMLengthProof];
        simp[List.mapM];
        simp[List.map];
        simp[List.toFindom];
        congr;
        funext x;
        apply x.elim0;
      }
      case CONS => {
        intros x n f1 IH;
        simp[Findom.mapM, Findom.map];
        simp[Findom.toList];
        simp[List.mapM];
        sorry
      }
}

#check Findom.mapM_map

def Tensor1D.map (v: Tensor1D) (f: Int -> Int): Tensor1D := (Tensor1D.mk v.size0 (v.data.map f))

def Tensor1D.mapM {M: Type -> Type} [Monad M] [LawfulMonad M]
  (v: Tensor1D) (f: (Int) → M (Int)):
  M Tensor1D := do
  let data <- Findom.mapM v.data f
  pure (Tensor1D.mk v.size0 data)


theorem Tensor1D.mapM_map [Monad M] [LawfulMonad M] (v: Tensor1D) (f: Int -> Int) (fM: Int → M Int)
      (F: forall a, fM a = pure (f a)):
    v.mapM fM = return v.map f := by sorry


def Tensor1D.mapMWithFlatIndex {M: Type -> Type} [Monad M] [LawfulMonad M]
  (v: Tensor1D) (f: Int × Fin v.size0 → M (Int)):
  M Tensor1D := do
  let data <- Findom.mapMWithIndex v.data f
  pure (Tensor1D.mk v.size0 data)




-- Old mapM theory: def Tensor1D.mapMWithFlatIndex {M: Type -> Type} [Monad M]
-- Old mapM theory:   (v: Tensor1D) (f: TensorFlatIndex v.size0 → (FinInt 32) → M (FinInt 32)):
-- Old mapM theory:   M Tensor1D := do
-- Old mapM theory:   let data <-
-- Old mapM theory:       (List.zipFlatIndex v.data).mapM (fun (val, ix) => f (v.h_data_size ▸ ix) val)
-- Old mapM theory:   let temp := Tensor1D.mk data.length data rfl
-- Old mapM theory:   return temp
-- Old mapM theory:
-- Old mapM theory: theorem List.mapM_loop_map [Monad M] [LawfulMonad M]
-- Old mapM theory:     (l: List α) (f: α → β) (fM: α → M β) (results: List β):
-- Old mapM theory:     (forall a, fM a = return f a) →
-- Old mapM theory:     List.mapM.loop fM l results = return results.reverse ++ l.map f := by
-- Old mapM theory:   intros h
-- Old mapM theory:   revert results
-- Old mapM theory:   induction l with
-- Old mapM theory:   | nil => intros results; simp [map]; rfl
-- Old mapM theory:   | cons a l ih =>
-- Old mapM theory:       intros results
-- Old mapM theory:       simp [mapM.loop, map, h, ih, reverse_cons, append_assoc]
-- Old mapM theory:
-- Old mapM theory: theorem List.mapM_map [Monad M] [LawfulMonad M] (l: List α) (f: α → β) (fM: α → M β):
-- Old mapM theory:     (forall a, fM a = return f a) →
-- Old mapM theory:     l.mapM fM = return l.map f := by
-- Old mapM theory:   apply List.mapM_loop_map
-- Old mapM theory:
-- Old mapM theory:
-- Old mapM theory: theorem Tensor1D.mapM_map [Monad M] [LawfulMonad M] v f (fM: _ → _ → M _):
-- Old mapM theory:     (forall flat_index val, fM flat_index val = return f flat_index val) →
-- Old mapM theory:     mapMWithFlatIndex v fM = return mapWithFlatIndex v f := by
-- Old mapM theory:   intros h
-- Old mapM theory:   unfold mapWithFlatIndex
-- Old mapM theory:   unfold mapMWithFlatIndex
-- Old mapM theory:   rw [List.mapM_map]
-- Old mapM theory:   . simp [v.h_data_size]; rfl
-- Old mapM theory:   . intros a; cases a; simp [h]
