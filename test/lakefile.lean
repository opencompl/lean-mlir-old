import Lake
open Lake DSL

package «TestCanonicalizer» {
  -- add configuration options here
  dependencies := #[{
    name := "MLIR",
    src := Source.path "../",
  }]
  supportInterpreter := true
  binRoot := `TestCanonicalizer
}
