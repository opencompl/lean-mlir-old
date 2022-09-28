import Lake
open Lake DSL

package «MLIR»

lean_lib MLIR

@[defaultTarget]
lean_exe «mlir» {
  root := `MLIR
  supportInterpreter := true
}
