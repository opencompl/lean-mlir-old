- `Lean/Parser/Extension.lean`: `runParserAttributeHooks` seems to be for extending the parser.

- How does CompCert use SSA?
- Talk to the MLIR C infrastructure to compile our rewrites to PDL.
- write a custom version of `by` that lives in my own grammar!
  Use the extensible version of `eval`. 
- Extensible InfoView ?
- Delaborator is `expr -> stx`. I can write a custom `mlirthing -> fmt`.
- I would add new attributes and use this to guide your own 
  delaboration. Have a custom attribute that I apply to functions of
  this type `mlirthing -> fmt`.
- Another option: Go to `expr`, write a custom DSL on top of the
  tactic DSL. Write a custom version of `by { ... }`, like `byMLIR { ... }`
  that allows a custom.
- Interact with `InfoTree`. Extend `InfoTree`. to get `fmt`.

