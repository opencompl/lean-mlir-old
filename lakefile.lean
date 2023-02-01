import Lake
open Lake DSL

package «MLIR»

lean_lib MLIR

@[default_target]
lean_exe «mlir» {
  root := `MLIR
  supportInterpreter := true
}

-- require «egg-tactic» from git  "https://github.com/opencompl/egg-tactic-code" @ "499ef2d"
require Mathlib from git "https://github.com/leanprover-community/mathlib4" @ "master"
