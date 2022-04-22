import Lake
open Lake DSL

package MLIRSemantics {
  binRoot := `MLIRSemantics
  dependencies := #[{
    name := "MLIR",
    src := Source.git "https://github.com/lephe/lean-mlir.git" "feb5e250ef4da51408ebdc75b6e7cbd24c96b73e",
  }]
}
