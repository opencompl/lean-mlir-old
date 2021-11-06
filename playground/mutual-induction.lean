-- Testing mutual induction in LEAN "by hand"

mutual
  inductive Expr
  | expr_int: Int -> Expr
  | expr_neg: Expr -> Expr
  -- | expr_fn_call: (name: String) -> (args: List Expr) -> Expr
  -- | expr_fn_defn: (name: String) -> (body: Block) -> Expr
  -- inductive Stmt 
  -- | stmt_expr: (lhs: Expr) -> Stmt
  -- | stmt_assign: (lhs: Expr) -> (rhs: Expr) -> Stmt

  inductive Block
  | block_empty: Block
  -- | block_seq: (lhs: Block) -> (rhs: Stmt) -> Block
end
open Expr

-- ⊢ {motive_1 : Expr → Sort u_1} →
--   {motive_2 : Block → Sort u_1} →
--     (t : Expr) 
--     → ((t : Expr) → Expr.below t → motive_1 t) 
--     → ((t : Block) → Block.below t → motive_2 t)
--     → motive_1 t
-- Expr.below : Expr → Sort (max 1 u_1)
#check Expr.below
#print Expr.below
#check Expr.brecOn
#print Expr.brecOn
-- open Stmt
-- open Block

def induction_expr_stmt
  (mot_expr: Expr -> Prop)
  -- (mot_stmt: Stmt -> Prop)
  (mot_block: Block -> Prop)
  -- EXPR
  (base_expr_int: forall (i: Int), mot_expr (expr_int i)) 
  (ind_expr_neg: forall (e: Expr) 
    (MOT_NEG: mot_expr e), mot_expr (expr_neg e)):
  -- (ind_expr_fn_defn: forall (name: String) (body: Block),
  --   mot_block body -> mot_expr (expr_fn_defn name body))
  -- STMT
  -- (base_stmt_expr: forall (e: Expr) (MOT: mot_expr e), mot_stmt (stmt_expr e))
  -- BLOCK
  -- (base_block: mot_block (block_empty))
  -- (ind_block: forall (b: Block) (s: Stmt), 
  --   mot_block b -> mot_stmt s -> mot_block (block_seq b s)):
  -- FINAL
    forall (e: Expr), mot_expr e :=
    fun e => 
      let x := e 
      let y := sorry  
      let z := sorry
      let out := @Expr.brecOn mot_expr mot_block x y z
      out
