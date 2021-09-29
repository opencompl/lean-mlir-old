# EMLIR: embedded MLIR in LEAN

This provides infrastructure for:

- A parser from MLIR generic into LEAN data structures.
- A pretty printer from LEAN data structures back into MLIR.
- Ability to write proofs over MLIR.

This research project explores:

- What default logics are useful, and how best to enable them for MLIR? Hoare logic? Separation logic?
- Purely functional, immutable rewriter with a carefully chosen set of
  primitives to enable both reasoning and efficient rewriting.

# Build instructions

```
$ make
/home/bollu/work/2-mlir-verif$ make 
/home/bollu//work/lean4-contrib/build/stage1/bin/lean --version
Lean (version 4.0.0, commit 850fd84e4340, Release)
/home/bollu//work/lean4-contrib/build/stage1/bin/lean mlir.lean -c mlir-opt.c
/home/bollu//work/lean4-contrib/build/stage1/bin/leanc mlir-opt.c -o mlir-opt
```

