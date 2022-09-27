import Lake
open Lake DSL

package «dialect-projection»
lean_lib DialectProjection

@[defaultTarget]
lean_exe «dialect-projection» {
  root := `Main
  supportInterpreter := true
}
