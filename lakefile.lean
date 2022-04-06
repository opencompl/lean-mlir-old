import Lake
open Lake DSL

package MLIRSemantics {
  binRoot := `MLIRSemantics
  dependencies := #[{
    name := "MLIR",
    src := Source.git "https://github.com/lephe/lean-mlir.git" "9efb35eda4b75c4d04bb992ae1aca7215293a87c",
  }]
}
