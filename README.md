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
$ lake build
$ ./build/bin/MLIR <path-to-generic-mlir-file.mlir>
```

## Nix

Start a "lean shell" per the [Nix Setup documentation](https://leanprover.github.io/lean4/doc/setup.html#nix-setup) for lean:

> After installing (any version of) Nix (https://nixos.org/download.html), you can easily open a shell with the particular pre-release version of Nix needed by and tested with our setup (called the "Lean shell" from here on):
> ```
> $ nix-shell https://github.com/leanprover/lean4/archive/master.tar.gz -A nix
> ```

Ensure that these extra options are active in the nix you are running. Under NixOS, this means putting this in top-level config. Under a non OS level isntall, this means putting this in the `.nix-profile` and restarting the nix daemon.

```nix=
  nix = {
      package = pkgs.nixUnstable; # or versioned attributes like nix_2_4
      extraOptions = ''
        experimental-features = nix-command flakes
        max-jobs = auto  # Allow building multiple derivations in parallel
        keep-outputs = true  # Do not garbage-collect build time-only dependencies (e.g. clang)
        # Allow fetching build results from the Lean Cachix cache
        trusted-substituters = https://lean4.cachix.org/
        trusted-public-keys = cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY= lean4.cachix.org-1:mawtxSxcaiWE24xCXXgh3qnvlTkyU7evRRnGeAhD4Wk=
        allow-import-from-derivation = true
      '';
  };
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

# License

Some source code from Mathlib4 is included in `MLIR/Util/Mathlib4`, licenser under [Apache 2.0](https://github.com/leanprover-community/mathlib4/blob/master/LICENSE).
