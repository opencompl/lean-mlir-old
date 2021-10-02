# `mlir-lean`: embedded MLIR in LEAN

This provides infrastructure for:

- An embedding of the MLIR AST in lean (`MLIR/AST.lean`)
- An lightweight parser combinator library with error tracking (`MLIR/P.lean`)
- A parser from MLIR generic into LEAN data structures (`MLIR/MLIRParser.lean`)
- A embedded-domain-specific language to build MLIR generic operations via macros (`MLIR/EDSL.lean`)
- Ability to write proofs over MLIR.

```lean
def opRgnAttr0 : Op := (mlir_op_call%
 "module"() (
 {
  ^entry:
   "func"() (
    {
     ^bb0(%arg0:i 32, %arg1:i 32):
      %zero = "std.addi"(%arg0 , %arg1) : (i 32, i 32) -> i 32
      "std.return"(%zero) : (i 32) -> ()
    }){sym_name = "add", type = (i 32, i 32) -> i 32} : () -> ()
   "module_terminator"() : () -> ()
 }) : () -> ()
)
#print opRgnAttr0
```

As a research project, we explore:

- How to provide formal semantics for MLIR, especially in the presence of multiple dialects.
- What default logics are useful, and how best to enable them for MLIR? Hoare logic? Separation logic?
- Purely functional, immutable rewriter with a carefully chosen set of
  primitives to enable reasoning and efficient rewriting.

# Build instructions

```
$ leanpkg build bin
$ ./build/bin/MLIR <path-to-generic-mlir-file.mlir>
```

