## Directory structure


```
.
├── build -- build directory
├── docs
│   └── README.md -- this file
├── examples -- cutom examples
├── leanpkg.toml -- LEAN package manager file
├── LLVM_COMMIT -- hardcoded LLVM commit for CI
├── makefile -- common targets to `build`, `debug`, `test`.
├── MLIR
│   ├── Dialects
│   │   ├── Builtin.lean -- Encoding of MLIR's `std` dialect via EDSL
│   │   ├── Linalg.lean -- Encoding of MLIR's `linalg` dialect via EDSL
│   │   └── PDL.lean -- Encoding of MLIR's `pdl` dialect via EDSL
│   ├── Examples
│   │     TODO
│   ├── AST.lean -- Core AST data structure.
│   ├── CParser.lean -- ANSI C Parser. (EXPERIMENTAL)
│   ├── Doc.lean -- Lightweight pretty printer.
│   ├── EDSL.lean -- Macro embedding of MLIR-generic syntax.
│   ├── FFI.lean -- Foreign Function Interface to call MLIR. (EXPERIMENTAL)
│   ├── MLIRParser.lean -- MLIR parser built from P.lean
│   ├── P.lean -- lightweight text parser.
│   ├── PatternMatch.lean -- Tactics to build pattern matchers in LEAN.
│   └── TAST.lean -- Typed AST with a richer structure than AST.
├── MLIR.lean -- toplevel module file.
├── NOTES.md -- research notes
├── playground -- experiments
├── reading -- experiments
├── README.md -- toplevel readme
└── utils
    ├── build-llvm.sh
    └── README
```
