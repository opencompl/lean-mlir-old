---
title: "lean-mlir: Builtin dialect"
---

[Back to index](index.html)

Dialect URL: [https://mlir.llvm.org/docs/Dialects/Builtin/](https://mlir.llvm.org/docs/Dialects/Builtin/)

## Types

* Floating-point types are really hard to get right especially in formal reasoning. We leave that discussion to a future integration with a Lean floating-point library.
* "Floating-point or integer scalar" ([1]): we take this as `i*`, `si*`, `ui*` or `f*`
* We set `index` to be of arbitrary precision (Lean `Int`)

**Signless integers**

We interpret all integers as bitvectors of appropriate size. Signless "integers" remain simple bitvectors, while signed integers (`si*`) and unsigned integers (`ui*`) have a Lean interpretation as numerical values (`Int`).


[1]: https://mlir.llvm.org/docs/Dialects/Builtin/#complextype
