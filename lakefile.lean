import Lake
open Lake DSL

package MLIRSemantics {
  binRoot := `MLIRSemantics
  dependencies := #[{
    name := "MLIR",
    src := Source.git "https://github.com/lephe/lean-mlir.git" "3a3d71390b1dff65606e2e167d1cdc01d8304397",
  }]
}
