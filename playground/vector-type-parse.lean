-- === VECTOR TYPE ===
-- TODO: where is vector type syntax defined?
-- | TODO: fix bug that does not allow a trailing times.

-- static-dim-list ::= decimal-literal (`x` decimal-literal)*
declare_syntax_cat static_dim_list
syntax sepBy(numLit, "×") : static_dim_list

syntax "[static_dim_list|" static_dim_list "]" : term
macro_rules
| `([static_dim_list| $[ $ns:numLit ]×* ]) => do
      quoteMList ns.toList (<- `(Int))

def staticDimList0 : List Int := [static_dim_list| 1]
#reduce staticDimList0

def staticDimList1 : List Int := [static_dim_list| 1 × 2]
#reduce staticDimList1


-- vector-dim-list := (static-dim-list `x`)? (`[` static-dim-list `]` `x`)?
-- | WTF, the whole of vector-dim-list can be empty...
declare_syntax_cat vector_dim_list
syntax (static_dim_list "×")? ("[" static_dim_list "]" "×")? : vector_dim_list
-- vector-element-type ::= float-type | integer-type | index-type
-- vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
syntax "vector" "<" vector_dim_list mlir_type ">"  : mlir_type

set_option hygiene false in -- allow i to expand 
macro_rules
| `([mlir_type| vector < $[ $fixed?:static_dim_list × ]? $[ [ $scaled?:static_dim_list ] × ]? $t:mlir_type  >]) => do
      let fixedDims <- match fixed? with  
        | some s =>  `([static_dim_list| $s])
        | none => `((@List.nil Int))
      let scaledDims <- match scaled? with  
        | some s => `([static_dim_list| $s])
        | none => `((@List.nil Int))
      `(MLIRTy.vector [] $scaledDims [mlir_type| $t])

def vectorTy0 := [mlir_type| vector<i32>]
#print vectorTy0

def vectorTy1 := [mlir_type| vector<2 × i32>]
#print vectorTy0
