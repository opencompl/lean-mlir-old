import Lake
open Lake DSL

package MLIRSemantics {
  dependencies := #[{
    name := "MLIR",
    src := Source.git "https://github.com/opencompl/lean-mlir.git" "059b5d5922ad5b38bf30696afff0926edc6341bf",
  }]
}
