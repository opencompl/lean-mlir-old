/-
We try to use the theory of Trees That Grow [1] to encode
the generic tree shape of MLIR, and have dialects control the tree
nodes. This should ensure a uniform representation for all dialects
(as a tree that grows), and allow for typeclasses that represent
injections and projections into the generic tree.

However, the Lean kernel does not enjoy this encoding, where we have
a type family defined at the top-level.

[1] https://www.microsoft.com/en-us/research/uploads/prod/2016/11/trees-that-grow.pdf
-/
namespace ASTVarToplevel
/-
-- Functors that describe the shape of 'Op in Rgn' and 'Rgn in Op'
variable (opRgnF: Type -> Type) -- describe the shape of Rgn inside Op
variable (rgnOpF: Type -> Type) -- describe the shape of Op inside Rgn

mutual
inductive Op
| mk: String -> (opRgnF Region) -> Op

inductive OpList
| OneLop: Op -> OpList
| ConsOp: Op -> OpList  -> OpList

inductive Region
| mk: String -> (rgnOpF OpList) -> Region
end
-/
end ASTVarToplevel

/-
We try a different encoding of the same idea, this time making the type family
as arguments to the inductive types, instead of a top-level variable in
the mutual induction.
-/
namespace ASTArgInd
-- Functors that describe the shape of Op in Rgn and Rgn in Op
/-
mutual
-- (kernel) arg #4 of 'ASTArgInd.Op.mk' contains a non valid occurrence of the datatypes being declared
inductive Op (opRgnF: Type -> Type) (rgnOpF: Type -> Type)
| mk: String -> (opRgnF (Region opRgnF rgnOpF)) -> Op opRgnF rgnOpF

inductive Region (opRgnF: Type -> Type) (rgnOpF: Type -> Type)
| mk: String -> (rgnOpF (Op opRgnF rgnOpF)) -> Region opRgnF rgnOpF
end
-/
end ASTArgInd


/-
We try another encoding, where we allow the user to only 'enrich' the branches
of the inductives.

The type `TAG` is a `void*`, which allow us to attach data to different
branches of the inductive.

If we wish to disallow regions in ops, we could say `data Region = Void`. This would
make it impossible to construct such a region, as Void is uninhabited.
-/

namespace ASTGrowHask98
inductive TAG where
| Op: TAG
| Region: TAG

mutual
inductive Op (data: TAG -> Type) where
| mk: String -> Region data -> data OP -> Op data

inductive OpList (data: TAG -> Type) where
| OneLop: Op data -> OpList data
| ConsOp: Op data -> OpList data -> OpList data

inductive Region (data: TAG -> Type) where
| mk: String -> (data REGION) -> OpList data -> Region data
end
end ASTGrowHask98


namespace ASTGrowDependent
inductive TAG where
| OP: TAG
| REGION: TAG

@[reducible]
def TagData: TAG -> Type
| .OP => String
| .REGION => String × Int

def f: Σ(t: TAG), TagData t := ⟨TAG.OP, "foo"⟩

mutual
inductive Op (data: (Σ(t: TAG), TagData t) -> Type) where
| mk: (name: String) -> Region data -> data ⟨TAG.OP, name⟩ -> Op data

inductive OpList (data: (Σ(t: TAG), TagData t) -> Type) where
| OneLop: Op data -> OpList data
| ConsOp: Op data -> OpList data -> OpList data

-- I need induction-recursion here, to define OpList.length.
-- I can of course, alternatively, use List instead of OpList and just
-- use List.length, but this destroys my mutual inductive.

inductive Region (data: (Σ(t: TAG), TagData t) -> Type) where
| mk: (name: String) -> (ops: OpList data) ->  (data ⟨TAG.REGION, (name, 0)⟩) -> Region data
end
end ASTGrowDependent

namespace ASTGrowDependent2
inductive TAG where
| OP: TAG
| REGION: TAG

@[reducible]
def TagData: TAG -> Type
| .OP => String
| .REGION => String × Int

def f: Σ(t: TAG), TagData t := ⟨TAG.OP, "foo"⟩
/-
mutual
inductive Op (data: (Σ(t: TAG), TagData t) -> Type) where
| mk: (name: String) -> Region data -> data ⟨TAG.OP, name⟩ -> Op data

/-
application type mismatch
  List.length ops
argument has type
  _nested.List_1
but function has type
  List (Op data) → Nat

SID: I guess that the way in which this is implemented is that Lean inlines the definition of
    List inside the nested inductive block, which reduces this to the previous expression, where
    I cannot implement List.length directly.
-/
inductive Region (data: (Σ(t: TAG), TagData t) -> Type) where
| mk: (name: String) -> (ops: List (Op data)) ->  (data ⟨TAG.REGION, (name, List.length ops)⟩) -> Region data

end
-/
end ASTGrowDependent2

namespace ASTGrowDependent3
inductive TAG where
| OP: TAG
| REGION: TAG

@[reducible]
def TagData: TAG -> Type
| .OP => String
| .REGION => String × Int

def f: Σ(t: TAG), TagData t := ⟨TAG.OP, "foo"⟩

mutual
inductive Op (data: (Σ(t: TAG), TagData t) -> Type) where
| mk: (name: String) -> Region data -> data ⟨TAG.OP, name⟩ -> Op data

-- By tracking the information I need in the Op at the type level for the
-- end-user to prune the tree, I can expose this to the `data` field.
-- This now means that the user can choose to create a `data` function
-- which returns `Void` when the length is `>= 2`, or for any length.
-- This allows us to create "sub-ASTs" which can be embedded into larger
-- ASTs.
inductive OpList (data: (Σ(t: TAG), TagData t) -> Type): Nat -> Type where
| OneLop: Op data -> OpList data 1
| ConsOp: Op data -> OpList data n -> OpList data (n + 1)

inductive Region (data: (Σ(t: TAG), TagData t) -> Type) where
| mk: (name: String) -> (ops: OpList data n) ->  (data ⟨TAG.REGION, (name, n)⟩) -> Region data
end



end ASTGrowDependent3



namespace ASTGrowDependent3MLIR
/-
We make the above machinery more realistic by actually representing MLIR.
We use this to represent `Arith`, which has no regions, and `Scf.if`, which
has 2 regions, and `Scf.while, which has 1 region.

However, in this example, the limitation is that we do not handle extensible, parametric
MLIR types such as tensor<int<64>>, where int<?i> comes from `arith`, and `tensor<?t>` comes
from tensor, and `64` comes from the base MLIR language.
-/

inductive FinVec (t: Type): Nat -> Type where
| Nil: FinVec t 0
| Cons: t -> FinVec t n -> FinVec t (n + 1)

inductive TAG where
| OP: TAG
| BB: TAG
| REGION: TAG

@[reducible]
def TagData: TAG -> Type
| .OP => String × Nat × Nat-- name of op, number of args, number of regions
| .BB => String -- name of bb
| .REGION => String × Nat -- name of region, num instructions in region.


structure SSAVal where
  -- | TODO: figure out how to have parametric types.
  name: String

-- | TODO: Think about whether `data` should live in [Type] or in [Prop].
mutual
inductive Op (data: (Σ(t: TAG), TagData t) -> Prop) where
| mk: (name: String) -> (args: FinVec SSAVal as) -> (regions: FinVec (Region data) rs) -> data ⟨TAG.OP, (name, as, rs)⟩ -> Op data

inductive BasicBlock (data: (Σ (t: TAG), TagData t) -> Prop) where
| mk: (name: String) -> (ops: List (Op data)) -> data ⟨TAG.BB, name⟩ -> BasicBlock data

inductive Region (data: (Σ(t: TAG), TagData t) -> Prop) where
| mk: (name: String) -> (ops: FinVec (BasicBlock data) n) ->  (data ⟨TAG.REGION, (name, n)⟩) -> Region data
end

def RegionSingleBBCheck (s: Σ(t: TAG),TagData t): Prop :=
  match s with
  | ⟨.OP, _⟩ => True
  | ⟨.BB, _⟩ => True
  | ⟨.REGION, (_name, n)⟩ => if n == 1 then True else False

def TrueCheck (_s: Σ(t: TAG),TagData t): Prop := True

abbrev RegionSingleBB := Region RegionSingleBBCheck

abbrev BasicBlock' := BasicBlock TrueCheck

inductive Arith where
| Add: SSAVal -> SSAVal -> Arith
| Const: Int -> Arith

inductive Scf where
| If: Int -> BasicBlock' -> BasicBlock' -> Scf
| For: (val: SSAVal) -> (lo: SSAVal) -> (hi: SSAVal) -> BasicBlock' -> Scf

inductive Linalg where
| Generic:  RegionSingleBB -> Linalg

-- 1. Some inductive type that represents their syntax
-- 2. A projection from generic MLIR into their syntax, and
--    an embedding from their syntax into generic MLIR.
-- 3. An interpreter on their syntax (which is specified wrt our typeclass)
end ASTGrowDependent3MLIR
