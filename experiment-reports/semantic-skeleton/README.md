# Semantic skeleton

We split the problem into two parts: (1) the computable fragment , where we want lean to help us,
(2) the uncomputable / fixpointey fragment, where we want to use rewrite rules.

We split our semantics into two parts: the computable fragment is made up of regular lean `def`s.
The uncomputable fragment is made up of axiomatic rewrite rules. This mixture allows us to
reason about things like looping via rewrites, while retaining the ability to reason about regular
programs from within Lean's logic.

