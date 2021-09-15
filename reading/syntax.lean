import Lean.Parser
-- vim Lean/Parser/Syntax.lean
-- ~/work/lean4hack/stage1/bin/lean syntax.lean
-- src/Leanpkg/Toml.lean
-- Fileworker.lean
structure inst where
  name : String
  arg: String

structure MLIRModule where 
    name: String
    regions: String

declare_syntax_cat newstxcat
declare_syntax_cat newstxcatregion

syntax ident: newstxcatregion
syntax "{ " newstxcatregion  " }" : newstxcatregion

macro_rules
  | `(to_term $a:ident)  => `($a)
  | `(to_term ($a))      => `(to_term $a)

syntax "`[foo|" newstxcat "]" : term

-- src/Init/Notation.lean
-- macro_rules | `(tactic| trivial) => `(tactic| assumption)
macro_rules 
  | `(RE ER) => `("region")

macro_rules
  | `(XXmodule $name $rgn) => `(MLIRModule.mk $name $rgn)

macro_rules
  | `(XXX $m) => `($m)

macro_rules
  -- | `(@inst $name:ident ($as:strLit)) => `(inst.mk  $name.getId.toString $as)
  | `(@inst $name ($as:strLit)) => `(inst.mk  $name $as)

#check XXX Nat.add
#check @inst "Nat.add" ("foo")
#check (inst.mk "foo" "bar")

#check (RE   ER)
#check (MLIRModule.mk "mod" (RE ER))
#check (XXmodule "mod2" (RE ER))



inductive Value : Type where
  | str : String → Value
  | nat : Nat → Value
  | bool : Bool → Value
  | table : List (String × Value) → Value
  deriving Inhabited

def Value.lookup : Value → String → Option Value
  | Value.table cs, k => cs.lookup k
  | _, _ => none

-- TODO: custom whitespace and other inaccuracies
declare_syntax_cat val
syntax "True" : val
syntax "False" : val
syntax str : val
syntax num : val
syntax bareKey := ident  -- TODO
syntax key := bareKey <|> str
declare_syntax_cat keyCat @[keyCatParser] def key' := key  -- HACK: for the antiquotation
syntax keyVal := key " = " val
syntax table := "[" key "]" keyVal*
syntax inlineTable := "{" keyVal,* "}"
syntax inlineTable : val
syntax file := table*
declare_syntax_cat fileCat @[fileCatParser] def file' := file  -- HACK: for the antiquotation

open Lean

partial def ofSyntax : Syntax → Value
  | `(val|True) => Value.bool true
  | `(val|False) => Value.bool false
  | `(val|$s:strLit) => Value.str <| s.isStrLit?.get!
  | `(val|$n:numLit) => Value.nat <| n.isNatLit?.get!
  | `(val|{$[$keys:key = $values],*}) => toTable keys (values.map ofSyntax)
  | `(fileCat|$[[$keys] $kvss*]*) => toTable keys <| kvss.map fun kvs => ofSyntax <| Lean.Unhygienic.run `(val|{$kvs,*})
  | stx => unreachable!
  where
    toKey : Syntax → String
      | `(keyCat|$key:ident)  => key.getId.toString
      | `(keyCat|$key:strLit) => key.isStrLit?.get!
      | _                     => unreachable!
    toTable (keys : Array Syntax) (vals : Array Value) : Value :=
      Value.table <| Array.toList <| keys.zipWith vals fun k v => (toKey k, v)

open Lean.Parser

declare_syntax_cat foo

macro_rules
  | `(@toml $name:foo) => `($name)

#check(@toml "True")

-- def parse (input : String) : IO Value := do
--   -- HACKHACKHACK
--   let env ← importModules [{ module := `Leanpkg.Toml }] {}
--   let fileParser ← compileParserDescr (parserExtension.getState env).categories file { env := env, opts := {} }
--   let c := mkParserContext (mkInputContext input "") { env := env, options := {} }
--   let s := mkParserState input
--   let s := whitespace c s
--   let s := fileParser.fn c s
--   if s.hasError then
--     throw <| IO.userError (s.toErrorMsg c)
--   else if input.atEnd s.pos then
--     ofSyntax s.stxStack.back
--   else
--     throw <| IO.userError ((s.mkError "end of input").toErrorMsg c)
-- 
