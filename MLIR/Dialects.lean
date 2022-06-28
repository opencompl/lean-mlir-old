/-
## Interface for MLIR dialects
-/


/-
### Extended types and attributes

Dialects can define custom types and attributes that map to concrete Lean
datatypes satisfying the requirements of the type interface. These types and
attributes can be mixed together with other dialects' and used in proofs.

Because it is impossible to know in advance what set of dialects an operation
will be used with, most proofs should either (1) quantify on a dialect
interface and handle default cases, or (2) come with a lifting theorem that
performs this closure under context automatically.
-/

/-- The `DialectTypeIntf` typeclass defines the requirements for dialect-
    supplied custom types. These properties are available even on unknown
    custom types, which allows type-generic functions and proofs to be written.

    In the interface, `σ` is the type of "signatures" (parameters), while `ε`
    is the type of "extended values". For instance for `builtin.tensor`, `σ` is
    a datatype holding the dimensions and base type of the tensor, while `ε` is
    the type of tensor values (for any particular signature).

    Type families that quantify over `Type` or higher universes are not
    supported to avoid universe polymorphism in the core data structures; in
    this case, `ε` can be made to implement specific instances of the family
    instead. -/
class DialectTypeIntf (σ: Type) (ε: σ → Type): Type where

  /-- The type should be inhabited, so that SSA value accesses can return
      defaults instead of requiring a proof that every access is dominated by
      a definition (which is a very annoying statement to maintain). -/
  inhabited: forall (s: σ), ε s

  /-- The signature should have decidable equality, so we can compare types. -/
  typeEq: DecidableEq σ
  /-- The type should have decidable equality, so that rewriting tools can
      find matches involving concrete values (eg. in PDL).
      Note: This might be relaxed into a `BEq` in the future. -/
  eq: forall (s: σ), DecidableEq (ε s)

  /-- String representation of values in the type (eg. a multi-dimensional
      array for tensors).  -/
  str: (s: σ) → ε s → String
  /-- String representation of the type itself (eg. "tensor<4x4xf64>"). -/
  typeStr: σ → String

  -- TODO: DialectTypeIntf: Type signature, to match eg. "any builtin.vector"

def DialectTypeIntf.sigType {σ ε} (_: DialectTypeIntf σ ε): Type := σ
def DialectTypeIntf.extType {σ ε} (_: DialectTypeIntf σ ε) (s: σ): Type := ε s

-- Expressing typeclass instances this way helps resolution
instance {σ ε} [i: DialectTypeIntf σ ε] (s: σ): Inhabited (ε s) where
  default := i.inhabited s
instance {σ ε} [i: DialectTypeIntf σ ε] (s: σ): ToString (ε s) where
  toString := i.str s
instance {σ ε} [i: DialectTypeIntf σ ε]: DecidableEq σ :=
  i.typeEq
instance {σ ε} [i: DialectTypeIntf σ ε] (s: σ): DecidableEq (ε s) :=
  i.eq s


/-- The `DialectAttrIntf` typeclass defines the requirements for dialect-
   supplied custom attributes. -/
class DialectAttrIntf (α: Type) where

  /-- The attribute should have decidable equality so that rewriting can match
      against them (eg. in PDL). -/
  eq: DecidableEq α

  /-- String representation of attribute values. -/
  str: α → String

  -- TODO: More data on attributes

def DialectAttrIntf.type {α} (_: DialectAttrIntf α): Type := α

instance {α} [i: DialectAttrIntf α]: DecidableEq α := i.eq


/-
### Combinations of interfaces

As dialects can provide multiple sets of extended types and attributes (and
dialects may themselves be combined), the interface must allow for different
extensions to be combined.

The following instances allow extended types, attributes, and dialects to be
combined with `Sum`.
-/

-- Like Sum.rec, but not a recursor (hence supported for code generation)
@[reducible]
def Sum.cases {α β γ} (fα: α → γ) (fβ: β → γ): (α ⊕ β) → γ
  | .inl a => fα a
  | .inr b => fβ b

instance {σ₁ ε₁ σ₂ ε₂} [i₁: DialectTypeIntf σ₁ ε₁] [i₂: DialectTypeIntf σ₂ ε₂]:
    DialectTypeIntf (σ₁ ⊕ σ₂) (Sum.cases ε₁ ε₂) where
  inhabited s :=
    match s with
    | .inl s₁ => i₁.inhabited s₁
    | .inr s₂ => i₂.inhabited s₂
  typeEq := inferInstance
  eq s :=
    match s with
    | .inl s₁ => i₁.eq s₁
    | .inr s₂ => i₂.eq s₂
  str s :=
    match s with
    | .inl s₁ => i₁.str s₁
    | .inr s₂ => i₂.str s₂
  typeStr := Sum.cases i₁.typeStr i₂.typeStr

instance {α₁ α₂} [i₁: DialectAttrIntf α₁] [i₂: DialectAttrIntf α₂]:
    DialectAttrIntf (α₁ ⊕ α₂) where
  eq := inferInstance
  str := Sum.cases i₁.str i₂.str


/-
### Dialects
-/

-- TODO: Document and finish the Dialect interface
class Dialect (α σ) (ε: σ → Type): Type :=
  iα: DialectAttrIntf α
  iε: DialectTypeIntf σ ε

instance {α ε} [δ: Dialect α σ ε]: DialectAttrIntf α := δ.iα
instance {α ε} [δ: Dialect α σ ε]: DialectTypeIntf σ ε := δ.iε


/-
### Empty dialect

The empty dialect is a default value to start building hierarchies from. It is
used in a couple of aliases, eg. `MLIRTy` (for `MLIRType Dialect.empty`) and
`AttrVal` (for `AttrValue Dialect.empty`).
-/

inductive Void :=
deriving DecidableEq

instance: DialectTypeIntf Void (fun _ => Unit) where
  inhabited s := nomatch s
  typeEq      := inferInstance
  eq s        := nomatch s
  str s       := nomatch s
  typeStr s   := nomatch s

instance: DialectAttrIntf Void where
  eq          := inferInstance
  str a       := nomatch a

instance Dialect.empty: Dialect Void Void (fun _ => Unit) where
  iα := inferInstance
  iε := inferInstance


-- We write combinations of dialects with + as usual (no risk of confusion)
instance {α₁ σ₁ ε₁ α₂ σ₂ ε₂}:
  HAdd (Dialect α₁ σ₁ ε₁) (Dialect α₂ σ₂ ε₂)
       (Dialect (α₁ ⊕ α₂) (σ₁ ⊕ σ₂) (Sum.cases ε₁ ε₂)) where
  hAdd δ₁ δ₂ := {
    iα := inferInstance
    iε := inferInstance
  }

instance {α₁ σ₁ ε₁ α₂ σ₂ ε₂} [δ₁: Dialect α₁ σ₁ ε₁] [δ₂: Dialect α₂ σ₂ ε₂]:
    Dialect (α₁ ⊕ α₂) (σ₁ ⊕ σ₂) (Sum.cases ε₁ ε₂) :=
  δ₁ + δ₂


/-
### Coercions of dialects

The `CoeDialect` ckass is used to automatically inject individual dialects into
sums of dialects, which in turn allows automatic conversion of instances of
common MLIR data such as `MLIRType`, `AttrValue` and `Op` across dialects.
-/

class CoeDialect (δ₁: Dialect α₁ σ₁ ε₁) (δ₂: Dialect α₂ σ₂ ε₂) where
  coe_α: α₁ → α₂
  coe_σ: σ₁ → σ₂
  coe_ε: forall s, ε₁ s → ε₂ (coe_σ s)

instance (δ₁: Dialect α₁ σ₁ ε₁) (δ₂: Dialect α₂ σ₂ ε₂) [c: CoeDialect δ₁ δ₂]:
  Coe α₁ α₂ where coe := c.coe_α
instance (δ₁: Dialect α₁ σ₁ ε₁) (δ₂: Dialect α₂ σ₂ ε₂) [c: CoeDialect δ₁ δ₂]:
  Coe σ₁ σ₂ where coe := c.coe_σ
instance (δ₁: Dialect α₁ σ₁ ε₁) (δ₂: Dialect α₂ σ₂ ε₂) [c: CoeDialect δ₁ δ₂] s:
  Coe (ε₁ s) (ε₂ /-coe-/s) where coe := c.coe_ε s

instance (δ: Dialect α σ ε): CoeDialect δ δ where
  coe_α := id
  coe_σ := id
  coe_ε s := id

instance (δ₁: Dialect α₁ σ₁ ε₁) (δ₂: Dialect α₂ σ₂ ε₂):
    CoeDialect δ₁ (δ₁ + δ₂) where
  coe_α := .inl
  coe_σ := .inl
  coe_ε s := id

instance (δ₁: Dialect α₁ σ₁ ε₁) (δ₂: Dialect α₂ σ₂ ε₂):
    CoeDialect δ₂ (δ₁ + δ₂) where
  coe_α := .inr
  coe_σ := .inr
  coe_ε s := id

instance (δ: Dialect α σ ε): CoeDialect Dialect.empty δ where
  coe_α a := nomatch a
  coe_σ s := nomatch s
  coe_ε s := nomatch s
