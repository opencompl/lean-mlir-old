import MLIR.Dialects.Builtin
-- import MLIR.TAST
open MLIR.Doc
open MLIR.AST

-- =========
-- Our IR: lz.
-- We begin by defining an inductive type of our set of operation.
inductive lzop where
| lzthunkify
| lzforce
| lzconstruct
| lzcase
| lzap
| lzcall
| int
| lzfunc


/-
instance : Coe Nat Arity where
 coe n := Arity.nat n

def lzop_arg_arity: lzop -> (Arity × Arity)
| lzop.lzthunkify => (1, 1)
| lzop.lzforce => (1, 1)
| lzop.lzconstruct => (0, Arity.infty)
| lzop.lzcase => (1, 1)
| lzop.lzap => (1, Arity.infty) -- needs at least function.
| lzop.lzcall => (1, Arity.infty) -- need at least function.
| lzop.lzfunc => (0, 0)
| lzop.int => (1, 1)


def lzop_region_arity: lzop -> (Arity × Arity)
| lzop.lzcase => (1, Arity.infty)
| lzop.lzfunc => (1, 1)
| _ => (0, 0)

def lzop_bbargs_arity: lzop -> (Arity × Arity)
| _ => (0, 0)
-/


-- | Code generate pattern rewrite_antisym_sum as MLIR PDL
-- def pdl_rewrite_atisym_sum := codegen_pdl rewrite_antisym_sum
def main_end_to_end_lz: IO Unit := do
  IO.eprintln "HASKELL TEST\n=======\n"
  -- IO.eprintln matmul_ein
  -- IO.eprintln matmul_linalg
  -- IO.eprintln rewrite_antisym_sum
  -- IO.eprintln pdl_rewrite_antisym_sum



-- =========
-- Our surface syntax.
-- Let's write a minimal, untyped, lazy language dubbed 'huskell',
-- for it contains the husk of a haskell-like language.

declare_syntax_cat husk_constructor
declare_syntax_cat husk_lit
declare_syntax_cat husk_expr
-- | number
syntax num : husk_expr

-- | function application
syntax husk_expr husk_expr : husk_expr

-- | constructor
syntax husk_constructor husk_expr* : husk_expr

-- | function declaration
syntax "λ" husk_lit "->" husk_expr : husk_expr



declare_syntax_cat husk_case_arm
syntax "case" husk_expr "of" "{" sepBy(husk_case_arm, "|") "}" : husk_expr
declare_syntax_cat husk_case_lhs
syntax "|" husk_case_lhs "->" husk_expr : husk_case_arm

syntax num : husk_case_lhs
syntax husk_constructor (husk_lit <|> "_")* : husk_case_lhs


-- | toplevel definition
declare_syntax_cat husk_definition
syntax "defn" husk_lit ":="  : husk_definition

-- | toplevel consists of a sequence of definitions
declare_syntax_cat husk_toplevel
syntax "husk" "{" husk_definition* "}" : husk_toplevel

