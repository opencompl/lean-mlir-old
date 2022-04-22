import Lake
open Lake DSL

package MLIRSemantics {
  binRoot := `MLIRSemantics
  dependencies := #[{
    name := "MLIR",
    src := Source.git "https://github.com/lephe/lean-mlir.git" "5d29c1c1494e189cc1e519875534f52749fc1f5a",
  }]
}
