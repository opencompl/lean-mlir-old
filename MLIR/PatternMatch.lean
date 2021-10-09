import MLIR.AST

namespace MLIR.PatternMatch

abbrev lens (s: Type) (t: Type) (a: Type) (b: Type) := (s -> t) -> (a -> b)
abbrev lens' (s: Type) (t: Type) (a: Type) (b: Type) := lens s t s t
