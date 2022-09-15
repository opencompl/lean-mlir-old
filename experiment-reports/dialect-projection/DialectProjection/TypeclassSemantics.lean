/-
In this file, we explore the use of the tagless final style [1]
to encode SSA semantics.

[1] https://okmij.org/ftp/tagless-final/
-/
import Lean
open Lean
namespace Exp


-- https://okmij.org/ftp/tagless-final/course/lecture.pdf
inductive Exp where
| Lit: Int -> Exp
| Neg: Exp -> Exp
| Add: Exp -> Exp -> Exp

def Exp.eval: Exp -> Int
| .Lit i => i
| .Neg e => -1 * e.eval
| .Add e e' => e.eval + e'.eval

class ExpSYM (repr: Type) where
  lit: Int -> repr
  neg: repr -> repr
  add: repr -> repr -> repr

instance : ExpSYM Int where
  lit i := i
  neg i := (-i)
  add i j := i + j

instance : ExpSYM String where
  lit i := toString i
  neg i := s!"(neg {i})"
  add i i' := s!"(add {i} {i'})"
end Exp

namespace Tree
inductive Tree where
| Leaf: String -> Tree
| Node: String -> List Tree -> Tree
deriving BEq

open Exp

-- Serialize Exp into Tree

instance : ExpSYM Tree where
  lit n := .Node "Lit" [.Leaf (toString n)]
  neg e := .Node "Neg" [e]
  add e e' := .Node "Add" [e, e']


def fromTree {repr: Type} [ExpSYM repr] : Tree -> Except String repr
| .Node "Lit" [.Leaf n] => do
   Except.ok (ExpSYM.lit 42) -- TODO: convert from string to nat.
| .Node "Neg" [e] => do
       return (ExpSYM.neg (<- fromTree e))
| .Node "Add" [e, e'] => do
   return ExpSYM.add (<- fromTree e) (<-
fromTree e')
| _t => Except.error "incorrect tree"

end Tree

namespace PushNeg
open Exp

def Exp.pushNeg: Exp -> Exp
| .Lit v => .Lit v
| .Neg (.Lit v) => .Neg (.Lit v)
| .Neg (.Neg e) => Exp.pushNeg e
| .Neg (.Add e e') => .Add (Exp.pushNeg e) (Exp.pushNeg e')
| .Add e e' => .Add (Exp.pushNeg e) (Exp.pushNeg e')

inductive Ctx where
| Pos: Ctx
| Neg: Ctx

instance {repr: Type} [ExpSYM repr] : ExpSYM (Ctx -> repr) where
  lit n := fun ctx => match ctx with
    | .Pos => ExpSYM.lit n
    | .Neg => ExpSYM.neg (ExpSYM.lit n)
  neg e := fun ctx => match ctx with
    | .Pos => e .Neg
    | .Neg => e .Pos
  add e1 e2 := fun ctx =>  ExpSYM.add (e1 ctx) (e2 ctx)
end PushNeg

namespace HO -- higher order tagless final

class Symantics (repr: Type -> Type) where
  int : Int -> repr Int
  add: repr Int -> repr Int -> repr Int
  lam: (repr a -> repr b) -> repr (a -> b)
  app: repr (a -> b) -> repr a -> repr b

structure R (a: Type) where
  val : a

instance : Symantics R where
  int i := { val := i }
  add i j := { val := i.val + j.val }
  lam f := { val := fun a =>  (f (R.mk a)).val }
  app f a := R.mk $ f.val a.val

class BoolSYM (repr: Type -> Type) where
  bool: Bool -> repr Bool
  leq : repr Int -> repr Int -> repr Bool
  if_: repr Bool -> repr a -> repr a -> repr a

instance : BoolSYM R where
 bool b := R.mk b
 leq a a' := R.mk (a.val <= a'.val)
 if_ cond t e := R.mk $ if cond.val then t.val else e.val

class FixSYM (repr: Type -> Type) where
  fix: (repr a -> repr a) -> repr a

-- lol
partial instance : FixSYM R where
  fix := sorry


-- h is heaps

inductive IR (h: Type _ -> Type _): Type _ -> Type _ where
| int: Int -> IR h Int
| add: IR h t -> IR h t -> IR h t
| var: h t -> IR h t
-- | lam: (IR h t1 -> IR h t2) -> IR h (t1 -> t2) -- non-positive occurence, cannot be encoded in initial style!


end HO

namespace SSA
/-
inductive BB (repr: Type _ -> Type _ ): Type _ -> Type _ where
| entry: String -> BB repr a -> BB repr a -- begin a bb
| seq: BB repr a -> BB repr b -> BB repr b
| op: repr a -> BB repr a -- operation
| ret: repr a -> BB repr a -- only place where problem occurs.
| condbr: repr Bool -> String -> String -> BB repr Unit
| br: String -> BB repr Unit


class BBSemantics (repr: Type _ -> Type _) where
  bb: BB repr a -> repr a

structure R (a: Type) where
  val : a


instance : BBSemantics R where
  bb repr := match repr with
             | .entry name rest =>

-/

inductive Op: Type _ -> Type _ where
| add: Int -> Int -> Op Int
| lt: Int -> Int -> Op Bool
| const: Int -> Op Int

structure BBName where
  name: String

structure BBRef (a: Type _) where
  name: String

-- Terminator has single type for interprocedural control flow.
-- Inside and Outside
-- k for things that are unknown, in the grand CPS style
-- BB intra inter.
inductive Terminator: Type _ -> Type _ where
| br: BBRef i -> i -> Terminator Unit
| ret: o -> Terminator o
| condbr: Bool -> (BBRef i × i) -> (BBRef i' × i') -> Terminator Unit

-- BB has three two type: one for interprocedural control flow
-- one for intraprocedural control flow
-- Inside and Outside
-- BB intra inter.
-- BB <input-type> <interprocedural-out-type>
inductive BB: Type _ -> Type _ -> Type _ where
| begin: (i -> BB Unit o) -> BB i o
| seq: Op a -> (a -> BB Unit o) -> BB Unit o
| terminator: Terminator o -> BB Unit o

-- build a BB which takes 'Int' input, produces 'Int' output.
def prog0 : BB Int Int :=
  .begin (fun input =>
    .seq (.const 4) (fun j =>
    .seq (.add input j) (fun k =>
    .terminator (.ret k)
  )))

-- Build a region
-- The list of types is the labels that have been defined.
inductive RegionBuilder: List (Σ (i: Type), BBRef i) -> Type _ -> Type _ -> Type _ where
| lbl: ((ref: BBRef i) ->
   RegionBuilder (⟨ i, ref ⟩::ris) ri ro) -- if you want a label,
   ->  RegionBuilder ris ri ro -- I can then forget about the `i` and remember that the `is` have been defined
                                       -- you have an obligation to define it in the output
| define: (ref: BBRef i) -> BB i o -> RegionBuilder ris ri ro
    -> RegionBuilder (⟨i,ref⟩::ris) ri ro -- define defines an `i`.
| empty: RegionBuilder [] ri ro -- empty region defines no BBS.

def prog1: RegionBuilder [] Int Int :=
  .lbl (i := Int) (fun entry =>
     .define entry (.begin fun i =>
      .terminator (.ret i)
      ) .empty)



-- takes an int as input, produces an int as output
-- entry(input):
--   br loop (input, 0)
-- loop(i, k):
--   knew := k + 1
--   inew := i + 1
--   exit := knew == 10
--   condbr exit(inew), loop(inew, knew)
-- exit(inew):
--   ret inew
def prog2: RegionBuilder [] Int Int :=
  .lbl (i := Int) (fun entrybb =>
  .lbl (i := Int × Int) (fun loopbb =>
  .lbl (i := Int) (fun exitbb =>
     .define exitbb (.begin fun inew => .terminator (.ret inew)) $
     .define loopbb (.begin fun args =>
       .seq (.add 1 args.fst) (fun knew =>
       .seq (.add 1 args.snd) (fun inew =>
       .seq (.lt knew 10) (fun isExit =>
       -- .terminator (.ret knew)))) -- (.condbr isExit ⟨exitbb, inew⟩, ⟨loopbb, (inew, knew)⟩)))))
       .terminator (.condbr isExit (exitbb, inew) (loopbb, (inew, knew))))))
     ) $
     .define entrybb (.begin fun input =>
      .terminator (.br loopbb (input, 0))) $
     .empty)))

end SSA


namespace PartialEvaluator
-- Section 4.6
end PartialEvaluator
