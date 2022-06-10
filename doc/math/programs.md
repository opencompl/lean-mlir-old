---
title: "Featherweight MLIR: Programs"
---

[Back to index](../index.html)

## Introduction

We describe an extension of SSA that models three important concepts of MLIR.

1. **Nesting of CFGs**: MLIR adds the notion of *regions*, which are full CFGs passed as arguments to operations. Regions allow nested control flow to be represented faithfully in the familiar imperative way.
2. **Typing for SSA values**: MLIR associates types to SSA values, which allow the compiler to perform type checking and analysis.
3. **Extensions through dialects**: MLIR allows the set of programs to be extended by *dialects*, which add new data types and new operations.

We refer to this program model as "featherweight MLIR". Featherweight MLIR hides a number of practical details of MLIR, some of which are implemented in our Lean 4 framework. We argue in section TODO that our simplified model nonetheless captures the essential and novel concepts of MLIR.

> Aspects of MLIR that are not described in this abstraction but are supported in the framework include:
>
> * Attributes
> * Constraints
> * Multiple return values
> * Type signatures (rather than using just a set of new types)
> * Custom attributes
>
> Aspects that are not even supported in the framework include:
>
> * Variadic arguments
> * Operation and region traits

## Dialects and types

Let's first have a look at the main constructions for dialects. Because MLIR is an extensible language, the set of valid operations, types and ultimately programs depends on the dialects being in use. Most definitions are therefore parameterized by a set of dialects.

A dialect in featherweight MLIR is conventionnally written $δ = (\textsf{ops}, ε, =_ \epsilon, (\textsf{default}_ τ)_ {τ ∈ ε}, (=_ τ)_ {τ ∈ ε})$ and consists of:

* A set of additional operations;
* A set of additional types, $ε$ (for *extended*);
* A computable equality test on $ε$;
* A default value for each custom type $τ ∈ ε$, $\textsf{default}_ τ$;
* A computable equality test on concrete values of each type $τ$.

Dialects are pervasive in MLIR; for instance, we define the complete collection of SSA types available in a program written in dialect $δ$ by adding $ε$ to the language's built-in types:

$$
\begin{align*}
  \textsf{MLIRType}_ δ ::=
    &~ \textsf{int}~~signedness~~size \\
  | &~ \textsf{tuple}~~(τ\!s: \textsf{List MLIRType}_ δ) \\
  | &~ \textsf{fn}~~(from: \textsf{MLIRType}_ δ)~~(to: \textsf{MLIRType}_ δ) \\
  | &~ \textsf{extended}~~(τ: ε)~
\end{align*}
$$

Base types include integers (described later) and the empty tuple (unit type). Type constructors include tuples and functions, and the $\textsf{extended}$ constructor handles types provided by the dialect, by simply naming one of the types $τ ∈ ε$.^[One could define $\textsf{MLIRType}$ generically by making $δ$ an argument of the $\textsf{extended}$ constructor. But MLIR's extended types can quantify over types (eg. for containers), which would force the definition of $\textsf{MLIRType}$ to be universe polymorphic. Universe polymorphism adds complexity in the development process, and we argue that explicitly specifying the dialect is clearer for semantics.]

## Operations, basic blocks and regions

The three components of MLIR's flavour of SSA: operations, basic blocks, and regions, are mutually recursive. *Operations* are similar to instructions in LLVM; they are identified by a name, and take as parameters lists of SSA values (for computations) as well as basic block names (for branches). The new additions in Featherweight MLIR are the list of regions (described below) and a type. The type is always a function type; its input is a tuple indicating the types of the SSA arguments, and its output is the type of the operation's return value.

$$
\begin{align*}
  \textsf{SSAVal} ::=
    &~ \textsf{String} \\
  \textsf{Op}_ δ ::=
    &~ \textsf{mk\_op}~~(op\_name: \textsf{String}) \\
    &~ (args: \textsf{List SSAVal})~~(bb\_args: \textsf{List BBName}) \\
    &~ (regions: \textsf{List Region}_ δ)~~(type: \textsf{MLIRType}_ δ)~ \\
\end{align*}
$$

Operations are grouped sequentially to form basic blocks, which are named so they can be referred to by operations. The semantics of basic blocks correspond to the usual sequential flow of SSA. MLIR uses the block-argument variation of SSA; this is equivalent to using $ϕ$ nodes.

$$
\begin{align*}
  \textsf{BBName} ::=
    &~ \textsf{String} \\
  \textsf{BasicBlockStmt}_ δ :=
    &~ \textsf{stmt\_op}~~Op_ δ \\
  | &~ \textsf{stmt\_assign}~~\textsf{SSAVal}~~Op_ δ \\
  \textsf{BasicBlock}_ δ :=
    &~ \textsf{mk\_bb}~~(name: \textsf{BBName}) \\
    &~ (args: \textsf{List}~(\textsf{SSAVal} × \textsf{MLIRType}_ δ)) \\
    &~ (stmts: \textsf{List BasicBlockStmt}_ δ) \\
\end{align*}
$$

Finally, basic blocks are composed to form CFGs called regions.

$$
\begin{align*}
  \textsf{Region}_ δ :=
    &~ \textsf{mk\_region}~~(bbs: \textsf{List BasicBlock}_ δ)~
\end{align*}
$$

While IRs typically use a flat CFG, MLIR introduces nested control flow as a first-class concept to support abstractions that require control over scoping. This is done by passing sub-CGFs as region arguments to operations. Thus operations can express concepts that are intimately tied to control flow, such as loop bodies, branches of an if-statement, and case statements in functional programs.

> TODO: Insert an example that uses most of these notions. We can use MLIR syntax as long as the instructions don't use attributes


## Combinations and coercions of dialects

MLIR dialects are designed to each deal with a particular domain, such as tensor computations or control flow. As a consequence, any MLIR function will invariably use operations from different dialects. To model this, we define a monoidal operation to combine dialect interfaces; given

$$
δ_ 1 = (\textsf{op}_ 1, ε_ 1, =_ {ε_ 1}, (\textsf{default}_ τ)_ {τ ∈ ε_ 1}, (=_ τ)_ {τ ∈ ε_ 1}), \\
δ_ 2 = (\textsf{op}_ 2, ε_ 2, =_ {ε_ 2}, (\textsf{default}_ τ)_ {τ ∈ ε_ 2}, (=_ τ)_ {τ ∈ ε_ 2}),
$$

we build

$$
δ_ 1 + δ_ 2 = (\textsf{op}_ 1 ⊔ \textsf{op}_ 2, ε_ 1 ⊔ ε_ 2, =_ {ε_ 1 ⊔ ε_ 2}, (\textsf{default}_ τ)_ {τ ∈ ε_ 1 ⊔ ε_ 2}, (=_ τ)_ {τ ∈ ε_ 1 ⊔ ε_ 2}).
$$
Both computable equality tests and the default values are straightforward to define; in fact, Lean4's typeclass inference system is able to construct them automatically.

This operation is commutative and associative. It also has a natural unit: the empty dialect, which is of limited practical use (it has no operations) but serves as a useful default to start building hierarchies from.

$$
δ_ {empty} = (∅, ∅, =_ ∅, (), ())
$$

This construction forms a commutative monoid, so we refer to "combinations" of dialects without specifying the exact order.

In addition, there is a natural injection of dialects into sums, which we use to convert common MLIR types such as $\textsf{MLIRType}$ and $\textsf{Op}$ across dialects. This is implemented with Lean4's coercion framework, which essentially inserts the following conversions in code as needed:

* $δ →_ {coe} δ$
* $δ_ {empty} →_ {coe} δ$
* $δ_ 1 →_ {coe} δ_ 1 + δ_ 2$
* $δ_ 2 →_ {coe} δ_ 1 + δ_ 2$
