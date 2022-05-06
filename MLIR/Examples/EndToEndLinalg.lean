import MLIR.Dialects.Builtin
import MLIR.Dialects.Linalg

open MLIR.Doc
open MLIR.AST



-- End to end example of using the einstein summation convention frontend
-- to generate linalg code.

-- | define expressions via EDSL
def matmul_ein := [ein_factor| x_k_i x^i_l]

-- | Mix into MLIR definitions.
def matmul_linalg := [mlir_op|
   func @"main"(%a: tensor<3 × 4 × f32>, %b: tensor<3×4 ×  f32>, %out: tensor<3×4 × f32>)  {
      ^entry:
        a^i b_i (%out)
    }
]

#eval IO.eprintln $ Pretty.doc $  matmul_linalg

-- | pretty print as MLIR generic



-- | rewrite A + A^T = 0 when A is known anti-symmetric
-- | create a rewrite which rewrite %x + %x^T = 0 for known pattern.
-- | This shows compositionality of our rewrite system.
-- | we declare ops %x, %y as variable bindings.
-- def rewrite_antisym_sum (x: Pattern) := 
-- [rewrite| %x %y | [ein| x_ij + y_ij] -> [mlir_op| std.constant 0]]

-- | rewrite A^2 = A for known pattern A
-- | This shows how to capture variables when declaring rewrites.


-- | Code generate pattern rewrite_antisym_sum as MLIR PDL
-- def pdl_rewrite_atisym_sum := codegen_pdl rewrite_antisym_sum
def main_end_to_end_linalg: IO Unit := do
  IO.eprintln "LINALG TEST\n=======\n"
  -- IO.eprintln matmul_ein
  -- IO.eprintln matmul_linalg
  -- IO.eprintln rewrite_antisym_sum
  -- IO.eprintln pdl_rewrite_antisym_sum
