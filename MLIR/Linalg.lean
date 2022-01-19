import MLIR.AST
import MLIR.EDSL
import MLIR.Doc
import MLIR.StdDialect

open MLIR.AST
open MLIR.EDSL
open MLIR.Doc
-- open MLIR.StdDialect
open Std

-- LINALG
-- =====
-- Pretty syntax:
-- ==============
-- #matmul_accesses = [
--   affine_map<(m, n, k) -> (m, k)>,
--   affine_map<(m, n, k) -> (k, n)>,
--   affine_map<(m, n, k) -> (m, n)>
-- ]
-- 
-- #matmul_trait = {
--   doc = "C(m, n) += A(m, k) * B(k, n)",
--   indexing_maps = #matmul_accesses,
--   library_call = "linalg_matmul",
--   iterator_types = ["parallel", "parallel", "reduction"]
-- }
-- 
-- func @main(%A:memref<?x?xf32>, %B: memref<?x?xf32>, %C:memref<?x?xf32>) {
--   linalg.generic #matmul_trait
--     ins(%A, %B : memref<?x?xf32>,
--         memref<?x?xf32>)
--     outs(%C : memref<?x?xf32>)
--   {
--   ^bb0(%a: f32, %b: f32, %c: f32) :
--     %d = mulf %a, %b: f32
--     %e = addf %c, %d: f32
--     linalg.yield %e : f32
--   }
--   return
-- }

-- Generic syntax:
-- ===============
-- #map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
-- #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
-- #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
-- "module"() ( {
--   "func"() ( {
--   ^bb0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>):  // no predecessors
--     "linalg.generic"(%arg0, %arg1, %arg2) ( {
--     ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
--       %0 = "std.mulf"(%arg3, %arg4) : (f32, f32) -> f32
--       %1 = "std.addf"(%arg5, %0) : (f32, f32) -> f32
--       "linalg.yield"(%1) : (f32) -> ()
--     }) {  doc = "C(m, n) += A(m, k) * B(k, n)", 
--           indexing_maps = [#map0, #map1, #map2],
--           iterator_types = ["parallel", "parallel", "reduction"], 
--           library_call = "linalg_matmul", operand_segment_sizes = dense<[2, 1]> : vector<2xi32>
--        } : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
--     "std.return"() : () -> ()
--   }) {sym_name = "main", type = (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()} : () -> ()
--   "module_terminator"() : () -> ()
-- }) : () -> ()


namespace affine_syntax
 

end affine_syntax

namespace linalg
open affine_syntax
open MLIR.EDSL

-- https://mlir.llvm.org/docs/Dialects/Linalg/#linalggeneric-mlirlinalggenericop
declare_syntax_cat linalg_arglist 
-- declare_syntax_cat linalg_arglist_ops
-- declare_syntax_cat linalg_arglist_tys
-- syntax mlir_op_operand : linalg_arglist_ops
-- syntax mlir_op_type : linalg_arglist_tys
syntax  "(" sepBy(mlir_op_operand, ",") ":" 
  sepBy(mlir_op_operand, ",") ")"  : linalg_arglist

-- | TODO: create an MLIR trait attribute in parser
-- syntax "linalg.generic" mlir_attr
--   "ins" linalg_arglist
--   "outs" linalg_arglist
--   mlir_region : mlir_op -- linalg op

syntax "linalg.generic" mlir_attr_dict "ins" linalg_arglist "outs" linalg_arglist mlir_region: mlir_op

-- | TODO: to define this, we need to decide how attributes are implemented.
set_option hygiene false in -- need to disable hygiene for i32 expansion.
macro_rules
| `([mlir_op| linalg.generic $attrs ins ($invs,* : $intys,*) outs ($outvs,* : $outtys,*) $rgn]) => do
   let initList <- `([])
   let argsList <- (invs.getElems ++ outvs.getElems).foldlM (init := initList) fun xs x => `($xs ++ [[mlir_op_operand| $x]])
   `(Op.mk "linalg_generic" $argsList [] [[mlir_region| $rgn]] [mlir_attr_dict| $attrs] (MLIRTy.int 31))
    


#check [affine_map| affine_map<(x, y, z) -> (x, y)>]

#check [mlir_op|
   linalg.generic { 
      -- indexing_maps = [ affine_map<(m, n, k) -> (m, k)>,
      --   affine_map<(m, n, k) -> (k, n)>,
      --   affine_map<(m, n, k) -> (m, n)>
      -- ],                   
      library_call = "linalg_matmul",
      iterator_types = ["parallel", "parallel", "reduction"] 
   } ins(%A, %B : ) outs (%C :) {
     ^entry(%a: i32, %b: i32, %c: i32) :
       %d = mulf %a, %b: f32
       %e = addf %c, %d: f32
       -- linalg.yield %e : f32
   }
]

inductive iterator_type
| parallel 
| reduction

end linalg


-- EINSTEIN SUMMATION
-- ===================
namespace ns_einsum

inductive Ein
| Sym: String -> Ein
| Upper: Ein -> String -> Ein
| Lower: Ein -> String -> Ein
| Mul: Ein -> Ein -> Ein
| Add: Ein -> Ein -> Ein
| Sub: Ein -> Ein -> Ein

declare_syntax_cat ein_leaf

syntax ident : ein_leaf
syntax ein_leaf "^"  ident : ein_leaf
syntax "[ein_leaf|" ein_leaf "]" : term

macro_rules 
| `([ein_leaf| $xraw:ident ]) => do 
  let xstr := xraw.getId.toString
  let splits := xstr.split $ fun c => c == '_'
  match splits with 
  | x::xs => do 
    let fst <- `(Ein.Sym $(Lean.quote x))
    xs.foldlM (fun e ix => `(Ein.Lower $e $(Lean.quote ix))) fst
  | _ => `(Ein.Sym "will never reach ein_leaf")


macro_rules
| `([ein_leaf| $x:ein_leaf ^ $ixsraw:ident]) => do 
  let splits := ixsraw.getId.toString.split $ fun c => c == '_'
  match splits with 
  | ix::ixs => do
      let fst <- `(Ein.Upper [ein_leaf| $x] $(Lean.quote ix))
      ixs.foldlM (fun e ixcur => `(Ein.Lower $e $(Lean.quote ixcur))) fst
  | _ => `(Ein.Sym "will never reach ein_leaf")
  
def leaf0 : Ein := [ein_leaf| x ]
#print leaf0

def leafd : Ein := [ein_leaf| x ]
#print leafd

def leafu : Ein := [ein_leaf| x^j ]
#print leafu


def leafdd : Ein := [ein_leaf| x_i_j ]
#print leafdd

def leafdu : Ein := [ein_leaf| x_i^j ]
#print leafdu
-- 
def leafud : Ein := [ein_leaf| x^i_j ]
#print leafud
-- 
def leafuu : Ein := [ein_leaf| x^j^k ]
#print leafuu

-- 
declare_syntax_cat ein_factor
syntax ein_leaf : ein_factor
syntax ein_factor ein_leaf : ein_factor -- multiplication by juxtaposition
syntax "[ein_factor|" ein_factor "]"  : term

macro_rules
| `([ein_factor| $x:ein_leaf ]) => `([ein_leaf| $x])
| `([ein_factor| $x:ein_factor $y:ein_leaf]) => 
  `(Ein.Mul [ein_factor| $x]  [ein_leaf| $y])
  
def facu := [ein_factor| x^k]
#print facu

def facd := [ein_factor| x_k]
#print facd

def facuu := [ein_factor| x^k x^j]
#print facuu

def facud := [ein_factor| x^j x_j]
#print facud

def fac3 := [ein_factor| x^i_k y_j^k x_k_l]
#print fac3

declare_syntax_cat ein_term
syntax ein_factor : ein_term
syntax ein_term "+" ein_factor : ein_term
syntax ein_term "-" ein_factor : ein_term

syntax "[ein|" ein_term "]" : term

macro_rules 
| `([ein| $x:ein_factor ]) => `([ein_factor| $x ])
| `([ein| $x:ein_term + $y:ein_factor ]) => 
  `(Ein.Add [ein| $x ] [ein_factor| $y ])
| `([ein| $x:ein_term - $y:ein_factor ]) => 
  `(Ein.Sub [ein| $x ] [ein_factor| $y ])

def t0 : Ein := [ein| x_i ]
#print t0

def t1 : Ein := [ein| x_i x^k + y_j - z_i_k^l]
#print t1





-- syntax "(" ein_term ")" : ein_leaf
-- macro_rules -- | bracketed terms are leaves
-- | `([ein_leaf| ( $x:ein_term) ]) => `([ein| $x ])

-- def tbrack1 : Ein := [ein| (x + y)_j^k ]
-- #print tbrack1 
-- 
-- def tbrack2 : Ein := [ein| (x_j + y_j)^k_l ]
-- #print tbrack2 


-- | this is really only defined for factors.
def get_ixs_inorder(e: Ein): List String :=
match e with
| Ein.Sym _ => []
| Ein.Upper e ix => get_ixs_inorder e ++ [ix]
| Ein.Lower e ix => get_ixs_inorder e ++ [ix]
| Ein.Sub l r => get_ixs_inorder l ++ get_ixs_inorder r
| Ein.Add l r => get_ixs_inorder l ++ get_ixs_inorder r
| Ein.Mul l r => get_ixs_inorder l ++ get_ixs_inorder r

-- | get lower and upper indexes of einstein summation term.
def get_low_up_ixs(e: Ein): List String × List String := 
  match e with
  | Ein.Sym _ => ([], [])
  | Ein.Upper e ix => 
      let (low, up) := get_low_up_ixs e 
      (low, ix::up)
  | Ein.Lower e ix => 
      let (low, up) := get_low_up_ixs e 
      (ix::low, up)
  | Ein.Mul l r => 
      let (lowl, upl) := get_low_up_ixs l
      let (lowr, upr) := get_low_up_ixs r
      (lowl ++ lowr, upl ++ upr)
  | _ => ([], [])


def get_ein_sym (e: Ein): String := 
match e with
| Ein.Sym s => s
| Ein.Upper e _ => get_ein_sym e
| Ein.Lower e _ => get_ein_sym e
| _ => "UNK"

def codegen_ein_index (i: Int) (e: Ein): List (SSAVal × Op) × SSAVal × Int :=  
   let arr := SSAVal.SSAVal $ get_ein_sym e
   let ixs : List String := get_ixs_inorder e
   let outname := SSAVal.SSAVal $ "einix" ++ toString i
   let outop := Op.mk "load" ( [arr] ++ ixs.map (SSAVal.SSAVal ∘ toString)) [] [] (AttrDict.mk []) [mlir_type| i32]
   ([(outname, outop)], outname, i+1)

-- | assumption: once we see an upper/lower, we only have 
def codegen_ein_mul (i: Int) (e: Ein): List (SSAVal × Op) × SSAVal × Int := 
  match e with
  | Ein.Mul l r => 
       let (opsl, outl, i) := codegen_ein_mul i l
       let (opsr, outr, i) := codegen_ein_mul i r
       let outname := SSAVal.SSAVal ("v" ++ toString i)
       let outop := [mlir_op| "mul" ({{ outl }}, {{ outr }} ) : () ]
       (opsl ++ opsr ++ [(outname, outop)], outname, i+1)
  | _ => codegen_ein_index i e 


-- | codegen einstein summation output being stored into `out`.
def codegen_ein_loop_body (e: Ein) : Region := 
  let (ls, us) := get_low_up_ixs e
  -- | partition indexes into repeated and unrepeated indexes.
  -- | generate loop for each repeated index.
  -- | generate array indexing for each unrepated index.
  let (_, unrepeated):= List.partition (fun l => us.contains l) ls
  let unrepeated := List.eraseDups unrepeated
  let (body, rhsval, _) := codegen_ein_mul 1 e
  let body := body.map fun (name, val) => BasicBlockStmt.StmtAssign name val

  let unrepeated := unrepeated.map SSAVal.SSAVal 

  let lhs_prev_val := SSAVal.SSAVal "lhs_prev"
  let lhsval := SSAVal.SSAVal "lhs"
  let lhs_load := Op.mk "load" ([lhsval] ++ unrepeated) [] [] (AttrDict.mk []) [mlir_type| i32]
  let lhs_add := Op.mk "add" [lhs_prev_val, rhsval] [] [] (AttrDict.mk []) [mlir_type|i32]

  let lhs_store := Op.mk "store" ([lhsval] ++ unrepeated ++ [rhsval] ) [] [] (AttrDict.mk []) [mlir_type| i32]
  let body := body ++ [BasicBlockStmt.StmtAssign lhs_prev_val lhs_load, BasicBlockStmt.StmtAssign lhsval lhs_add, BasicBlockStmt.StmtOp lhs_store]

  Region.mk $ [BasicBlock.mk "entry" [] body]


partial def codegen_ein_loop_nest (e: Ein) : Op := 
  let (ls, us) := get_low_up_ixs e
  -- | partition indexes into repeated and unrepeated indexes.
  -- | generate loop for each repeated index.
  -- | generate array indexing for each unrepated index.
  let (repeated, unrepeated):= List.partition (fun l => us.contains l) ls
  let body := codegen_ein_loop_body e
  let rec go (ixs: List String) : Op := 
    match ixs with
    | [] => [mlir_op| "scf.execute_region" () (<[ body ]>) : ()]
    | ix::ixs => 
      let body : Op := go ixs
      Op.mk "for" [SSAVal.SSAVal ix] [] [Region.mk [BasicBlock.mk "entry" [] [BasicBlockStmt.StmtOp body ]]] (AttrDict.mk []) [mlir_type| ()]
  go $ List.eraseDups (repeated ++ unrepeated)


#eval IO.eprintln $ Pretty.doc $ codegen_ein_loop_nest [ein| x_i x^i]
#eval IO.eprintln $ Pretty.doc $ codegen_ein_loop_nest [ein| x_i x_i]
#eval IO.eprintln $ Pretty.doc $ codegen_ein_loop_nest [ein| x_i_j y^j_i_k]


syntax "[ein|" ein_term "]": mlir_op

def foo := [ein| x_i x^i]
#check foo

macro_rules
| `([mlir_op| [ein| $x:ein_term ] ]) => `(codegen_ein_loop_nest [ein| $x])

def einAsMlirOp0 := [mlir_op| "scf.while" (%x) ({ 
   ^entry: 
      addi %c0 , %x
--      -- | use einstein summation convention inside
--      -- the `[mlir_op|` DSL as we build a `scf.while` op:
      -- [ein| x_i_j x_j^k]
}) : ()
]

#eval IO.eprintln $ Pretty.doc $ einAsMlirOp0

-- https://pytorch.org/docs/stable/nn.html
-- torch has equations for everything!

-- https://www.tensorflow.org/api_docs/python/tf/nn

-- atrous convolution
-- ==================
-- output[batch, height, width, out_channel] =
--     sum_{dheight, dwidth, in_channel} (
--         filters[dheight, dwidth, in_channel, out_channel] *
--         value[batch, height + rate*dheight, width + rate*dwidth, in_channel]
--     )


-- tf.nn.avg_pool(
--     input, ksize, strides, padding, data_format=None, name=None
-- )
-- ============================
-- NO DESCRIPTION


-- mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td
-- ===========================================================
--     linalg.copy(%arg0, %arg1) : memref<?xf32, stride_specification>,
--                                 memref<?xf32, stride_specification>


-- pooling base:
--       output[x[0], ..., x[N-1]] =
--         REDUCE_{z[0], ..., z[N-1]}
--           input[
--                 x[0] * strides[0] - pad_before[0] + dilation_rate[0]*z[0],
--                 ...
--                 x[N-1]*strides[N-1] - pad_before[N-1] + dilation_rate[N-1]*z[N-1]
--                 ],
--     ```



-- conv op:
--       output[b, x[0], ..., x[N-1], k] =
--       sum_{z[0], ..., z[N-1], q}
--           filter[z[0], ..., z[N-1], q, k] *
--           padded_input[b,
--                        x[0] * strides[0] + dilation_rate[0] * z[0],
--                        ...,
--                        x[N-1] * strides[N-1] + dilation_rate[N-1] * z[N-1],
--                        q]
-- 

-- mlir/include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yaml
-- matvec, vecmat, matmul, batch_matmul
end ns_einsum


-- ultra high level way to defne rewrites: TableGen syntax -> PDL by MLIR
-- mid level: write PDL, and feed to MLIR [painful, you're now writing IR by hand]
-- low level: write C++ to define rewrites.
