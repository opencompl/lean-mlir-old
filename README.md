# `mlir-lean`: embedded MLIR in LEAN

This provides infrastructure for:

- An embedding of the MLIR AST in lean (`MLIR/AST.lean`)
- A lightweight pretty printer library to pretty print the MLIR AST and parse errors (`MLIR/Doc.lean`)
- A embedded-domain-specific language to build MLIR generic operations via macros (`MLIR/EDSL.lean`)
- A parser from MLIR generic into LEAN data structures (`MLIR/MLIRParser.lean`)
- A lightweight parser combinator library with error tracking (`MLIR/P.lean`)

```lean
def opRgnAttr0 : Op := (mlir_op_call%
 "module"() (
 {
  ^entry:
   "func"() (
    {
     ^bb0(%arg0:i32, %arg1:i32):
      %zero = "std.addi"(%arg0 , %arg1) : (i32, i32) -> i32
      "std.return"(%zero) : (i 32) -> ()
    }){sym_name = "add", type = (i32, i32) -> i32} : () -> ()
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

# Test instructions

```
$ cd examples; lit -v . # run all examples, report failures.
```

# Documentation

- Go to [`docs/README.md`](./docs/README.md) for library documentation.



# Other projects of interest

- [`tydeu/lean4-papyrus`](https://github.com/tydeu/lean4-papyrus) is an LLVM interface for Lean 4.
- [`GaloisInc/lean-llvm`](https://github.com/GaloisInc/lean-llvm) is Lean4 bindings to the LLVM library.
