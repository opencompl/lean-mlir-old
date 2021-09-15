# EMLIR: embedded MLIR in LEAN

This provides infrastructure for:

- A parser from MLIR generic into LEAN data structures.
- A pretty printer from LEAN data structures back into MLIR.
- Ability to write proofs over MLIR.

This research project explores:

- What default logics are useful, and how best to enable them for MLIR? Hoare logic? Separation logic?
- Purely functional, immutable rewriter with a carefully chosen set of
  primitives to enable both reasoning and efficient rewriting.

