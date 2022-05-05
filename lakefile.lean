import Lake
open Lake DSL

package MLIRSemantics {
  binRoot := `MLIRSemantics
  dependencies := #[{
    name := "MLIR",
    src := Source.git "https://github.com/opencompl/lean-mlir.git" "6693232a29eaa56437c16d2956197b09c92802ac",
  }]
}
