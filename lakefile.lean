import Lake
open Lake DSL

package MLIRSemantics {
  binRoot := `MLIRSemantics
  dependencies := #[{
    name := "MLIR",
    src := Source.git "https://github.com/lephe/lean-mlir.git" "26ffa804c78e824e0128fb54e27dfdfbc00d035b",
  }]
}
