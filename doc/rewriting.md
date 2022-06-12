# MLIR Semantics: Rewriting with PDL

We support parsing basic PDL rewrite specifications and generating theorems stating the correctness of the rewrites. This system can be used to efficiently prove that local rewrite rules are correct under context.

## Overview

The process of analyzing and proving a rewrite theorem is as follows:

1. The dialect definition provides *generic operation formats* as match terms, as well as *arbitrary constraints* which encode the invariants of the instructions of the dialect. Such invariants are implicit in the PDL pattern.
2. **Unification:** The PDL pattern is compiled to a set of match terms; all the operations in these terms are unified with an instance of their dialect-provided generic formats. This step essentially enriches the PDL pattern with invariants enforced by the dialect.
3. **Canonical match:** A *canonical match* of the PDL pattern is computed; this is a program of minismum size that matches the pattern. (Such a program may not be unique.)
4. **Canonical output**: The rewrite part of the PDL pattern is applied to the canonical match, which defines the *canonical output*.
5. **Statement generation**: A statement of semantic preservation is generated for the canonical match. It quantifies over any remaining unification variables (usually values, types, tensor dimensions...), assumes any *arbitrary constraints* required on operations by the dialect, then states equality of the semantics of the canonical match and canonical output.
6. **Simple proof:** The user then proves the generated statement, from which it follows that the rewrite is correct when applied to the canonical match.
7. **Extensions under context:** Depending on the properties of the dialect, the proof can then be automatically extended to show that the rewrite is correct on *more* inputs; such extensions include interleaving unrelated statements (which rely on SSA invariants) or shuffling the order of instructions (which may rely on the absence of side-effects).

## Pattern matching on MLIR programs

We define a simple type of first-order match terms on MLIR programs. The motivation for introducing this type rather than using PDL directly is that we also want to use it to characterize constraints on instructions and perform unification, which is impractical when using PDL.

<span style='text-decoration:underline'>*Match terms*</span>

Match terms are defined by the following grammar:

```
value  ::= ValueVar STRING                // written "$STRING"
         | ValueConst TYPE VALUE
type   ::= TypeVar STRING                 // written "!STRING"
         | TypeConst TYPE
op     ::= OpKnown STRING (arg*) (ret*)
arg    ::= value ":" type
ret    ::= value ":" type

TODO: attributes
```

Using the same variable name with two different sorts of variables is possible but confusing. The translation from PDL always generates names that are unique across sorts.

Note how there is no recursive structure. Instead, interesting matching patterns rely on using a *combination* of terms that share variables.

<span style='text-decoration:underline'>*Substitutions*</span>

A substitution is a function that maps value variables and type variables to SSA value names and MLIR types. More specifically, it is a pair of functions:

* `σ_v: String → SSAVal`
* `σ_t: String → MLIRType δ`

<span style='text-decoration:underline'>*Instances*</span>

An MLIR operation is said to be an instance of an `op` match term

## Formalization of a PDL subset

<span style='text-decoration:underline'>*Matching pattern*</span>

A PDL program (as specified in a `pdl.pattern` operation)

<span style='text-decoration:underline'>*Matching operations*</span>

**`pdl.operand`**

```
[#1] pdl.operand
[#2] pdl.operand: <TYPE>
[#3] pdl.operand: %var
```

Defines a new

## Unification

