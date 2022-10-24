import Lake
open Lake DSL

package «MLIR»

lean_lib MLIR

@[default_target]
lean_exe «mlir» {
  root := `MLIR
  supportInterpreter := true
}

require «egg-tactic» from git  "https://github.com/opencompl/egg-tactic-code" @ "a493294"
require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "e094621836b13a341f006ca926e4e24ed09a9c1c"
require std from git "https://github.com/leanprover/std4.git" @ "d1d14d616938a07210b11d78165421a9a9163e48"
require Qq from git "https://github.com/gebner/quote4.git" @ "5be427045b61c9d40f9f4b46b3ade98e019a1ad1"
