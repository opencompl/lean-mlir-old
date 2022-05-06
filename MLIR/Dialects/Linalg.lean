import MLIR.AST
import MLIR.EDSL
import MLIR.Doc
import MLIR.Dialects.Builtin

open MLIR.AST
open MLIR.EDSL
open MLIR.Doc
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

syntax "linalg.yield" mlir_op_operand ":" mlir_type : mlir_op

macro_rules
| `([mlir_op| linalg.yield $arg : $ty]) =>
    `([mlir_op| "linalg.yield" ($arg) : ($ty) -> ()])

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
   let tysList <- (intys.getElems ++ outtys.getElems).foldlM (init := initList) fun xs x => `($xs ++ [[mlir_type| $x]])
   `(Op.mk "linalg_generic" 
        $argsList 
        []
        [[mlir_region| $rgn]]
        [mlir_attr_dict| $attrs] (MLIRTy.fn (MLIRTy.tuple $tysList) (MLIRTy.tuple [])))
    


#check [affine_map| affine_map<(x, y, z) -> (x, y)>]


def linalgGeneric0 :=  [mlir_op|
   linalg.generic { 
       indexing_maps = [ affine_map<(m, n, k) -> (m, k)>,
         affine_map<(m, n, k) -> (k, n)>,
         affine_map<(m, n, k) -> (m, n)>
      ],                   
      library_call = "linalg_matmul",
      iterator_types = ["parallel", "parallel", "reduction"] 
   } ins(%A, %B : ) outs (%C :) {
     ^entry(%a: i32, %b: i32, %c: i32) :
       %d = mulf %a, %b: f32
       %e = addf %c, %d: f32
       linalg.yield %e : f32
   }
]

#eval IO.eprintln $ Pretty.doc $ linalgGeneric0


end linalg


-- EINSTEIN SUMMATION
-- ===================
namespace ns_einsum

inductive EinLeaf
| Sym: String -> EinLeaf
| Upper: EinLeaf -> String -> EinLeaf
| Lower: EinLeaf -> String -> EinLeaf

inductive EinFactor
| Mul: EinLeaf -> EinLeaf -> EinFactor

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
    let fst <- `(EinLeaf.Sym $(Lean.quote x))
    xs.foldlM (fun e ix => `(EinLeaf.Lower $e $(Lean.quote ix))) fst
  | _ => `(EinLeaf.Sym "will never reach ein_leaf")


macro_rules
| `([ein_leaf| $x:ein_leaf ^ $ixsraw:ident]) => do 
  let splits := ixsraw.getId.toString.split $ fun c => c == '_'
  match splits with 
  | ix::ixs => do
      let fst <- `(EinLeaf.Upper [ein_leaf| $x] $(Lean.quote ix))
      ixs.foldlM (fun e ixcur => `(EinLeaf.Lower $e $(Lean.quote ixcur))) fst
  | _ => `(Ein.Sym "will never reach ein_leaf")
  
def leaf0 : EinLeaf := [ein_leaf| x]
#print leaf0

def leafd : EinLeaf := [ein_leaf| x ]
#print leafd

def leafu : EinLeaf := [ein_leaf| x^j ]
#print leafu


def leafdd : EinLeaf := [ein_leaf| x_i_j ]
#print leafdd

def leafdu : EinLeaf := [ein_leaf| x_i^j ]
#print leafdu
-- 
def leafud : EinLeaf := [ein_leaf| x^i_j ]
#print leafud
-- 
def leafuu : EinLeaf := [ein_leaf| x^j^k ]
#print leafuu

-- 
declare_syntax_cat ein_factor
syntax ein_leaf : ein_factor
syntax ein_factor ein_leaf : ein_factor -- multiplication by juxtaposition
syntax "[ein_factor|" ein_factor "]"  : term

macro_rules
| `([ein_factor| $x:ein_leaf ]) => `([ein_leaf| $x])
| `([ein_factor| $x:ein_leaf $y:ein_leaf]) => 
  `(EinFactor.Mul [ein_leaf| $x]  [ein_leaf| $y])
  
def facu := [ein_factor| x^k]
#print facu

def facd := [ein_factor| x_k]
#print facd

def facuu := [ein_factor| x^k x^j]
#print facuu

def facud := [ein_factor| x^j x_j]
#print facud

-- syntax "(" ein_term ")" : ein_leaf
-- macro_rules -- | bracketed terms are leaves
-- | `([ein_leaf| ( $x:ein_term) ]) => `([ein| $x ])

-- def tbrack1 : Ein := [ein| (x + y)_j^k ]
-- #print tbrack1 
-- 
-- def tbrack2 : Ein := [ein| (x_j + y_j)^k_l ]
-- #print tbrack2 


-- | this is really only defined for factors.
def EinLeaf.get_ixs_inorder(e: EinLeaf): List String :=
match e with
| EinLeaf.Sym _ => []
| EinLeaf.Upper e ix => get_ixs_inorder e ++ [ix]
| EinLeaf.Lower e ix => get_ixs_inorder e ++ [ix]

-- | get lower and upper indexes of einstein summation term.
def EinLeaf.get_low_up_ixs(e: EinLeaf): List String × List String := 
  match e with
  | EinLeaf.Sym _ => ([], [])
  | EinLeaf.Upper e ix => 
      let (low, up) := get_low_up_ixs e 
      (low, ix::up)
  | EinLeaf.Lower e ix => 
      let (low, up) := get_low_up_ixs e 
      (ix::low, up)

def EinFactor.get_low_up_ixs(e: EinFactor): List String × List String :=
match e with
| EinFactor.Mul x y => 
    let (l, u) := EinLeaf.get_low_up_ixs x
    let (l', u') := EinLeaf.get_low_up_ixs y
    (l ++ l', u ++ u')


def EinLeaf.get_sym (e: EinLeaf): String := 
match e with
| EinLeaf.Sym s => s
| EinLeaf.Upper e _ => get_sym e
| EinLeaf.Lower e _ => get_sym e


def EinFactor.left (e: EinFactor): EinLeaf :=
  match e with
  | EinFactor.Mul l r => l

 def EinFactor.right (e: EinFactor): EinLeaf :=
  match e with
  | EinFactor.Mul l r => r


-- | safe way to construct iterator types.
    inductive iterator_types := 
    | parallel | reduction

    macro_rules
    | `([mlir_attr_val| iterator_types.parallel]) => `(AttrVal.str "parallel")
    | `([mlir_attr_val| iterator_types.reduction]) => `(AttrVal.str "reduction")
      

partial def EinFactor.codegen (e: EinFactor) (out: SSAVal)  : Op := 
  let (ls, us) := EinFactor.get_low_up_ixs e
  -- | partition indexes into repeated and unrepeated indexes.
  -- | generate loop for each repeated index.
  -- | generate array indexing for each unrepated index.
  let (repeated, _):= List.partition (fun ix => us.contains ix) ls
  let repeated := List.eraseDups  repeated
  
  let iteration_vars := ls ++ us
  let iteration_vars := List.eraseDups iteration_vars

  let unrepeated := iteration_vars.filter (fun ix => not (repeated.contains ix))

  -- | full input space consists of each iteration variable
  let input_tuple : AffineTuple := AffineTuple.mk (iteration_vars.map (AffineExpr.Var))
  let output_tuple : AffineTuple := AffineTuple.mk (unrepeated.map (AffineExpr.Var))
  let leaf0_tuple := AffineTuple.mk $ (EinLeaf.get_ixs_inorder (EinFactor.left e)).map (AffineExpr.Var)
  let leaf1_tuple :=  AffineTuple.mk $ (EinLeaf.get_ixs_inorder (EinFactor.right e)).map (AffineExpr.Var)
  let rgn := [mlir_region| {
    ^entry(%a : f32, %b : f32, %c: f32):
      %mul = mulf %a, %b : f32
      %out = addf %c, %mul : f32
      linalg.yield %out : f32
  }]

  let indexing_maps := 
    AttrVal.list [
                   AttrVal.affine (AffineMap.mk input_tuple leaf0_tuple)
                 , AttrVal.affine (AffineMap.mk input_tuple leaf1_tuple)
                 , AttrVal.affine (AffineMap.mk input_tuple output_tuple)]
  let attrdict := [mlir_attr_dict| { 
      indexing_maps = $(indexing_maps),
      library_call = "linalg_matmul",
      iterator_types = [iterator_types.parallel,
                        iterator_types.parallel,
                        iterator_types.reduction] 
      }] -- input iter1 , input access 2, output access

  let leaf0_arg := SSAVal.SSAVal $ EinLeaf.get_sym (EinFactor.left e)
  let leaf1_arg := SSAVal.SSAVal $ EinLeaf.get_sym (EinFactor.right e)
  (Op.mk "linalg_generic" [leaf0_arg, leaf1_arg, out] [] [rgn] attrdict (MLIRTy.int 31))
  

syntax ein_factor "(" mlir_op_operand ")" : mlir_op

macro_rules
| `([mlir_op|   $e:ein_factor ( $rand:mlir_op_operand  ) ]) => `(EinFactor.codegen [ein_factor| $e]  [mlir_op_operand| $rand ])

def einAsMlirOp0 := [mlir_op| "scf.while" (%x) ({ 
   ^entry: 
      x_i x^ik (%y)
--      -- | use einstein summation convention inside
--      -- the `[mlir_op|` DSL as we build a `scf.while` op:
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
