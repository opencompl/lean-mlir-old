import Lake
open Lake DSL

package «MLIR» {
  -- add configuration options here
  supportInterpreter := true
  libName := "MLIR"
  binRoot := `MLIR
  libRoots := #[`MLIR] 
}
