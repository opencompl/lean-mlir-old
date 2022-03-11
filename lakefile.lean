import Lake
open Lake DSL

package MLIRSemantics {
  binRoot := `MLIRSemantics
  dependencies := #[{
    name := "MLIR",
    src := Source.git "https://github.com/opencompl/lean-mlir.git" "e29d0850e943198d50b691aeed7b6eec0be202df",
  }]
}
