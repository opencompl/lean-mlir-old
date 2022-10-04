import Lake
open Lake DSL

require ITree from "ITree"

package «MLIR» {
  -- add configuration options here
  supportInterpreter := true
  libName := "MLIR"
  binRoot := `MLIR
  libRoots := #[`MLIR] 
}
