### Theory
- The notion of `scoped` to ensure that our variables and bindings are
   properly scoped when we push/pop out of regions.
- Cleaning up the codebase to remove the dialect parameters for extention types.

### Arith:
- Connecting to decision procedure written by chris
- Writing a tactic that uses the decision procedure to prove equality.
- Checking that all the rewrites in Hacker's delight pass verification
  (Chris has done this. We need to integrate it into the larger framework).
   
### SCF
- `scf.for`, `scf.if`.
- Loop peeling, constant if folding, adjacent loop fusion.

### Linalg+Vector
- Operations: `generic1d`, `generic2d`, `insert`, `extract`, `fill`, `transpose`, `permute`.
- Rewrite in terms of `mathlib` objects.
- Loop reordering is **not done**, but we claim we do it (very bad! I should NOT have
  done this). Lean is very slow on these theorems.

### Peephole correctness
- A way to talk about locations in the IR.
- A lifting theorem, to lift local rewrites into globally correct rewrites.
  What is the correct theorem statement for this?
