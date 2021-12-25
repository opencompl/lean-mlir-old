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
│   ├── AST.lean -- Core AST data structure.
│   ├── CombBasic.lean -- Encoding of CIRCT's Comb dialect via EDSL.
│   ├── Doc.lean -- Lightweight pretty printer.
│   ├── EDSL.lean -- Macro embedding of MLIR-generic syntax.
│   ├── Main.lean -- Main file of executable that exposes our MLIR generic parser.
│   ├── MLIRParser.lean -- MLIR parser built from P.lean
│   ├── PatternMatch.lean -- Tactics to build pattern matchers in LEAN.
│   ├── P.lean -- lightweight text parser.
│   ├── Semantics.lean -- Monadic semantics.
│   └── StdDialect.lean -- Encoding of MLIR's `std` dialect via EDSL
├── MLIR.lean -- toplevel module file.
├── NOTES.md -- research notes
├── playground -- experiments
├── reading -- experiments
├── README.md -- toplevel readme
└── utils
    ├── build-llvm.sh
    └── README
```
