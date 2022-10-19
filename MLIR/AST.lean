import MLIR.Doc
import MLIR.Dialects
open Lean PrettyPrinter

open MLIR.Doc
open Pretty -- open typeclass for `doc`

namespace MLIR.AST

-- Affine expressions [TODO: find some way to separate this out]
-- ==================
inductive AffineExpr
| Var: String -> AffineExpr

instance : Pretty AffineExpr where
  doc e := match e with
  | AffineExpr.Var v => doc v

deriving instance DecidableEq for AffineExpr

inductive AffineTuple
| mk: List AffineExpr -> AffineTuple

instance : Pretty AffineTuple where
  doc t := match t with
  | AffineTuple.mk es => [doc| "(" (es),*  ")"]

deriving instance DecidableEq for AffineTuple

inductive AffineMap
| mk: AffineTuple -> AffineTuple -> AffineMap

 instance : Pretty AffineMap where
  doc t := match t with
  | AffineMap.mk xs ys => doc xs ++ " -> " ++ doc ys

deriving instance DecidableEq for AffineMap



-- EMBEDDING
-- ==========

inductive BBName
| mk: String -> BBName

instance : Pretty BBName where
  doc name := match name with
              | BBName.mk s => [doc| "^" s]

deriving instance DecidableEq for BBName

inductive Dimension
| Known: Nat -> Dimension
| Unknown: Dimension

deriving instance DecidableEq for Dimension


inductive SSAVal : Type where
  | SSAVal : String -> SSAVal
deriving DecidableEq

def SSAValToString (s: SSAVal): String :=
  match s with
  | SSAVal.SSAVal str => str

instance : ToString SSAVal where
  toString := SSAValToString

inductive TensorElem :=
| int: Int -> TensorElem
| float: Float -> TensorElem
| bool: Bool -> TensorElem
| nested: List TensorElem -> TensorElem
| empty: TensorElem

inductive Signedness :=
| Signless -- i*
| Unsigned -- u*
| Signed   -- si*
deriving DecidableEq

-- TODO: make this depend only on `code`
inductive MLIRType (ty: Type) :=
| int: Signedness -> Nat -> MLIRType ty
| float: Nat -> MLIRType ty
| tensor1d: MLIRType ty -- tensor of int values.
| tensor2d: MLIRType ty -- tensor of int values.
| tensor4d: MLIRType ty -- tensor of int values.
| index:  MLIRType ty
| undefined: String → MLIRType ty
| extended: ty → MLIRType ty
| erased: MLIRType ty -- A type that is erased by dialect retraction.

-- We define "MLIRTy" to be just the basic types outside of any dialect
abbrev MLIRTy := @MLIRType Void
-- Other useful abbreviations
abbrev MLIRType.i1: MLIRType ty := MLIRType.int .Signless 1
abbrev MLIRType.i32: MLIRType ty := MLIRType.int .Signless 32

-- An SSA value with a type
abbrev TypedSSAVal (ty: Type) := SSAVal × MLIRType ty

mutual

-- | TODO: factor Symbol out from AttrValue
inductive AttrValue (attr: Type) :=
| symbol: String -> AttrValue attr -- symbol ref attr
| str : String -> AttrValue attr
| int : Int -> AttrValue attr
| nat: Nat -> AttrValue attr
| bool : Bool -> AttrValue attr
| float : Float -> AttrValue attr
| affine: AffineMap -> AttrValue attr
| permutation: List Nat -> AttrValue attr -- a permutation
| list: List (AttrValue attr) -> AttrValue attr
-- | guaranteee: both components will be AttrValue attr.Symbol.
-- | TODO: factor symbols out.
| nestedsymbol: AttrValue attr -> AttrValue attr -> AttrValue attr
| alias: String -> AttrValue attr
| dict: AttrDict attr -> AttrValue attr
| opaque_: (dialect: String) -> (value: String) -> AttrValue attr
| opaqueElements: (dialect: String) -> (value: String) -> (type: MLIRType attr) -> AttrValue attr
| unit: AttrValue attr
| extended: attr → AttrValue attr
| erased: AttrValue attr

-- https://mlir.llvm.org/docs/LangRef/#attributes
-- | TODO: add support for mutually inductive records / structures
inductive AttrEntry (attr: Type) :=
  | mk: (key: String)
      -> (value: AttrValue attr)
      -> AttrEntry attr
-- TODO: make this depend only on `attr`.
inductive AttrDict (attr: Type) :=
| mk: List (AttrEntry attr) -> AttrDict attr

end

-- We define "AttrVal" to be just the basic attributes outside of any dialect
abbrev AttrVal := @AttrValue Void


mutual
-- | TODO: make this `record` when mutual records are allowed?
-- | TODO: make these arguments optional?
-- TODO: make this depend only on `attr`, `type`.
inductive Op  (attr: Type) (ty: Type) where
 | mk: (name: String)
      -> (res: List (TypedSSAVal ty))
      -> (args: List (TypedSSAVal ty))
      -> (regions: List (Region attr ty))
      -> (attrs: AttrDict attr)
      -> Op attr ty

inductive Region (attr: Type) (ty: Type) where
| mk: (name: String)
      -> (args: List (TypedSSAVal ty))
      -> (ops: List (Op attr ty)) -> Region attr ty
end

-- Attribute definition on the form #<name> = <val>
inductive AttrDefn (attr: Type) where
| mk: (name: String) -> (val: AttrValue attr) -> AttrDefn attr

inductive Module (ty: Type) (attr: Type) where
| mk: (functions: List (Op ty attr))
      -> (attrs: List (AttrDefn attr))
      ->  Module ty attr


def Op.name: Op attr ty -> String
| Op.mk name .. => name

def Op.res: Op attr ty -> List (TypedSSAVal ty)
| Op.mk _ res .. => res

def Op.resNames: Op attr ty → List SSAVal
| Op.mk _ res .. => res.map Prod.fst

def Op.resTypes: Op attr ty → List (MLIRType ty)
| Op.mk _ res .. => res.map Prod.snd

def Op.args: Op attr ty -> List (TypedSSAVal ty)
| Op.mk _ _ args .. => args

def Op.argNames: Op attr ty → List SSAVal
| Op.mk _ _ args .. => args.map Prod.fst

def Op.argTypes: Op attr ty → List (MLIRType ty)
| Op.mk _ _ args .. => args.map Prod.snd

def Op.regions: Op attr ty -> List (Region attr ty)
| Op.mk _ _ _ regions _ => regions

def Op.attrs: Op attr ty -> AttrDict attr
| Op.mk _ _ _ _ attrs => attrs


instance: Coe String SSAVal where
  coe (s: String) := SSAVal.SSAVal s

instance : Coe Int TensorElem where
  coe (i: Int) := TensorElem.int i

instance : Coe  (List Int) TensorElem where
  coe (xs: List Int) := TensorElem.nested (xs.map TensorElem.int)

instance : Coe String (AttrValue δ) where
  coe (s: String) := AttrValue.str s

instance : Coe Int (AttrValue δ) where
  coe (i: Int) := AttrValue.int i

instance : Coe (String × AttrValue δ) (AttrEntry δ) where
  coe (v: String × AttrValue δ) := AttrEntry.mk v.fst v.snd

instance : Coe  (AttrEntry δ) (String × AttrValue δ) where
  coe (v: AttrEntry δ) :=
  match v with
  | AttrEntry.mk key val => (key, val)

instance : Coe (List (AttrEntry δ)) (AttrDict δ) where
  coe (v: List (AttrEntry δ)) := AttrDict.mk v

 instance : Coe (AttrDict δ) (List (AttrEntry δ)) where
  coe (v: AttrDict δ) := match v with | AttrDict.mk as => as


def Region.name (region: Region attr ty): BBName :=
  match region with
  | Region.mk name args ops => BBName.mk name

def Region.ops (region: Region attr ty): List (Op attr ty) :=
  match region with
  | Region.mk name args ops => ops


-- Coercions across dialects
/-
variable [δ₁: Dialect attr₁ σ₁ ε₁] [δ₂: Dialect attr₂ σ₂ ε₂] [c: CoeDialect δ₁ δ₂]
-/

/-
def coeMLIRType [ty: Code ty] [ty': Code ty'] [inj: InjectCode ty ty']:
 MLIRType ty  → MLIRType ty'
  | .int sgn n   => .int sgn n
  | .float n     => .float n
  | .index       => .index
  | .undefined n => .undefined n
  | .tensor1d => .tensor1d
  | .tensor2d => .tensor2d
  | .tensor4d => .tensor4d
  | .erased => .erased
  | .extended s  => .extended (inj.injectCode ty ty' s)


instance {ty: Code ty} {ty': Code ty'} [inj: InjectCode ty ty']:
    Coe (MLIRType ty) (MLIRType ty') where
  coe := coeMLIRType

mutual
variable [δ₁: Dialect attr₁ σ₁ ε₁] [δ₂: Dialect attr₂ σ₂ ε₂] [c: CoeDialect δ₁ δ₂]

private def coeAttrValue: AttrValue δ₁ → AttrValue δ₂
  | .nat n => .nat n
  | .symbol s => .symbol s
  | .permutation p => .permutation p
  | .str s => .str s
  | .int i => .int i
  | .bool b => .bool b
  | .float f  => .float f
  | .affine map => .affine map
  | .list l => .list (coeAttrValueList l)
  | .nestedsymbol a₁ a₂ => .nestedsymbol (coeAttrValue a₁) (coeAttrValue a₂)
  | .alias s => .alias s
  | .dict d => .dict (coeAttrDict d)
  | .opaque_ d v => .opaque_ d v
  | .opaqueElements d v ty => .opaqueElements d v ty
  | .unit => .unit
  | .extended a => .extended (c.coe_attr _ _ a)
  | .erased => .erased

private def coeAttrValueList: List (AttrValue δ₁) → List (AttrValue δ₂)
  | [] => []
  | v :: values => coeAttrValue v :: coeAttrValueList values

private def coeAttrEntry: AttrEntry δ₁ → AttrEntry δ₂
  | .mk key value => .mk key (coeAttrValue value)

private def coeAttrEntryList: List (AttrEntry δ₁) → List (AttrEntry δ₂)
  | [] => []
  | e :: entries => coeAttrEntry e :: coeAttrEntryList entries

private def coeAttrDict: AttrDict δ₁ → AttrDict δ₂
  | .mk entries => .mk <| coeAttrEntryList entries
end

instance {δ₁: Dialect attr₁ σ₁ ε₁} {δ₂: Dialect attr₂ σ₂ ε₂} [CoeDialect δ₁ δ₂]:
    Coe (AttrValue δ₁) (AttrValue δ₂) where
  coe := coeAttrValue

instance {δ₁: Dialect attr₁ σ₁ ε₁} {δ₂: Dialect attr₂ σ₂ ε₂} [CoeDialect δ₁ δ₂]:
    Coe (AttrEntry δ₁) (AttrEntry δ₂) where
  coe := coeAttrEntry

instance {δ₁: Dialect attr₁ σ₁ ε₁} {δ₂: Dialect attr₂ σ₂ ε₂} [CoeDialect δ₁ δ₂]:
    Coe (AttrDict δ₁) (AttrDict δ₂) where
  coe := coeAttrDict

mutual
variable [δ₁: Dialect attr₁ σ₁ ε₁] [δ₂: Dialect attr₂ σ₂ ε₂] [c: CoeDialect δ₁ δ₂]

def coeOp: Op δ₁ → Op δ₂
  | .mk name res args  regions attrs =>
      .mk name res args  (coeRegionList regions) (Coe.coe attrs)

def coeOpList:
    List (Op δ₁) → List (Op δ₂)
  | [] => []
  | s :: ops => coeOp s :: coeOpList ops

def coeRegion: Region δ₁ → Region δ₂
  | .mk name args ops => .mk name args (coeOpList ops)

def coeRegionList: List (Region δ₁) → List (Region δ₂)
  | [] => []
  | r :: rs => coeRegion r :: coeRegionList rs

end

instance {δ₁: Dialect attr₁ σ₁ ε₁} {δ₂: Dialect attr₂ σ₂ ε₂} [CoeDialect δ₁ δ₂]:
    Coe (Op δ₁) (Op δ₂) where
  coe := coeOp

instance {δ₁: Dialect attr₁ σ₁ ε₁} {δ₂: Dialect attr₂ σ₂ ε₂} [CoeDialect δ₁ δ₂]:
    Coe (List (Op δ₁)) (List (Op δ₂)) where
  coe := coeOpList

instance {δ₁: Dialect attr₁ σ₁ ε₁} {δ₂: Dialect attr₂ σ₂ ε₂} [CoeDialect δ₁ δ₂]:
    Coe (Region δ₁) (Region δ₂) where
  coe := coeRegion

instance {δ₁: Dialect attr₁ σ₁ ε₁} {δ₂: Dialect attr₂ σ₂ ε₂} [CoeDialect δ₁ δ₂]:
    Coe (List (Region δ₁)) (List (Region δ₂)) where
  coe := coeRegionList

instance {δ₁: Dialect attr₁ σ₁ ε₁} {δ₂: Dialect attr₂ σ₂ ε₂} [CoeDialect δ₁ δ₂]:
    Coe (Region δ₁) (Region δ₂) where
  coe := coeRegion

instance {δ₁: Dialect attr₁ σ₁ ε₁} {δ₂: Dialect attr₂ σ₂ ε₂} [CoeDialect δ₁ δ₂]:
    Coe (List (Region δ₁)) (List (Region δ₂)) where
  coe := coeRegionList
-/


instance : Pretty Signedness where
  doc sgn :=
    match sgn with
    | .Signless => "Signless"
    | .Unsigned => "Unsigned"
    | .Signed => "Signed"

instance : Pretty Dimension where
  doc dim :=
  match dim with
  | Dimension.Unknown => "?"
  | Dimension.Known i => doc i

partial instance : Pretty TensorElem where
  doc (t: TensorElem) :=
    let rec go (t: TensorElem) :=
      match t with
       | TensorElem.int i => doc i
       | TensorElem.bool b => doc b
       | TensorElem.float f => doc f
       | TensorElem.nested ts => [doc| "["  (ts.map go),* "]" ]
       | TensorElem.empty => ""
    go t

-- | TODO: allow typeclass instances inside mutual blocks
section
variable {ty: Type} [ct: Code ty]

partial def docMLIRType: MLIRType ty → Doc
  | .int .Signless k => [doc| "i"k]
  | .int .Unsigned k => [doc| "u"k]
  | .int .Signed k => [doc| "si"k]
  | .float k => [doc| "f"k]
  | .tensor1d => [doc| "tensor1d"]
  | .tensor2d => [doc| "tensor2d"]
  | .tensor4d => [doc| "tensor4d"]
  | .index => [doc| "index"]
  | .undefined name => [doc| "!" name]
  | .erased => [doc| "erased"]
  | .extended sig =>  ct.showCode sig
end

mutual
partial def docAttrVal [ca: Code attr]: AttrValue attr → Doc
  | .symbol s => "@" ++ doc_surround_dbl_quot s
  | .permutation ps => [doc| "[permutation " (ps),* "]"]
  | .nestedsymbol s t => (docAttrVal s) ++ "::" ++ (docAttrVal t)
  | .str str => doc_surround_dbl_quot str
  | .int i => doc i ++ ":" ++ "i64"
  | .nat i => doc i ++ " : " ++ "index"
  | .bool b => if b then "true" else "false"
  | .float f => doc f ++ ":" ++ "f64"
  | .affine aff => "affine_map<" ++ doc aff ++ ">"
  | .list xs => "[" ++ Doc.Nest (vintercalate_doc (xs.map docAttrVal) ", ") ++ "]"
  | .alias a => "#" ++ a
  | .dict d => docAttrDict d
  | .opaque_ dialect val => [doc| "#" (dialect) "<"  (val) ">"]
  | .opaqueElements dialect val ty => [doc| "#opaque<" (dialect) ","  (val) ">" ":" (docMLIRType ty)]
  | .unit => "()"
  | .extended a => ca.showCode a
  | .erased => "<erased>"

partial def docAttrEntry [ca: Code attr]: AttrEntry attr → Doc
  | .mk k v => k ++ " = " ++ (docAttrVal v)

partial def docAttrDict [ca: Code attr]: AttrDict attr → Doc
  | .mk attrs =>
      if List.isEmpty attrs
      then Doc.Text ""
      else "{" ++ Doc.Nest (vintercalate_doc (attrs.map docAttrEntry)  ", ")  ++ "}"
end

instance [Code ty] : Pretty (MLIRType ty) where
 doc := docMLIRType

instance [Code attr] : Pretty (AttrValue attr) where
 doc := docAttrVal

instance  [Code attr]: Pretty (AttrEntry attr) where
  doc := docAttrEntry

instance [Code attr]: Pretty (AttrDict attr) where
   doc := docAttrDict

instance [Code attr]: Pretty (AttrDefn attr) where
  doc (v: AttrDefn attr) :=
  match v with
  | AttrDefn.mk name val => "#" ++ name ++ " := " ++ (doc val)

instance : Pretty SSAVal where
   doc (val: SSAVal) :=
     match val with
     | SSAVal.SSAVal name => Doc.Text ("%" ++ name)

instance : ToFormat SSAVal where
    format (x: SSAVal) := layout80col (doc x)

-- | TODO: allow mutual definition of typeclass instances. This code
-- | would be so much nicer if I could pretend that these had real instances.
mutual

def op_to_doc [Code attr] [Code ty] (op: Op attr ty): Doc :=
    match op with
    | (Op.mk name res args rgns attrs) =>
        /- v3: macros + if stuff-/
        [doc|
          "\"" name "\""
          "(" (op.argNames),* ")"
          (ifdoc rgns.isEmpty then "" else "(" (nest (list_rgn_to_doc rgns);*) ")")
          attrs]

        /- v2: macros, but no if stuff
        [doc|
          "\"" name "\""
          "(" (args),* ")"
          (if bbs.isEmpty then [doc| ""] else [doc| "[" (bbs),* "]"])
          (if rgns.isEmpty then [doc| ""] else[doc| "(" (nest rgns.map rgn_to_doc);* ")"])
          attrs ":" ty] -/

        /- v1: no macros
        let doc_name := doc_surround_dbl_quot name
        let doc_bbs := if bbs.isEmpty
                       then doc ""
                       else "[" ++ intercalate_doc bbs ", " ++ "]"
        let doc_rgns :=
            if rgns.isEmpty
            then Doc.Text ""
            else " (" ++ nest_vgroup (rgns.map rgn_to_doc) ++ ")"
        let doc_args := "(" ++ intercalate_doc args ", " ++ ")"

        doc_name ++ doc_args ++  doc_bbs ++ doc_rgns ++ doc attrs ++ " : " ++ doc ty -/

def list_op_to_doc [Code attr] [Code ty]: List (Op attr ty) → List Doc
  | [] => []
  | op :: ops => op_to_doc op :: list_op_to_doc ops

-- | TODO: fix the dugly syntax
def rgn_to_doc [Code attr] [Code ty]: Region attr ty → Doc
  | (Region.mk name args ops) =>
    [doc| {
        (ifdoc args.isEmpty
         then  "^" name ":"
         else  "^" name "(" (args.map $ fun (v, t) => [doc| v ":" t]),* ")" ":");
        (nest list_op_to_doc ops);* ; } ]

def list_rgn_to_doc [Code attr] [Code ty]: List (Region attr ty) → List Doc
  | [] => []
  | r :: rs => rgn_to_doc r :: list_rgn_to_doc rs
end

instance [Code attr] [Code ty]: Pretty (Op attr ty) where
  doc := op_to_doc

instance [Code attr] [Code ty]: Pretty (Region attr ty) where
  doc := rgn_to_doc

instance [Code attr] [Code ty]: Pretty (Region attr ty) where
  doc := rgn_to_doc

def AttrEntry.key (a: AttrEntry δ): String :=
match a with
| AttrEntry.mk k v => k

def AttrEntry.value (a: AttrEntry δ): AttrValue δ :=
match a with
| AttrEntry.mk k v => v


def AttrDict.empty : AttrDict attr := AttrDict.mk []

def Op.empty (name: String) : Op attr ty := Op.mk name [] [] [] AttrDict.empty

-- | TODO: needs to happen in a monad to ensure that ty has the right type!
def Op.addArg (o: Op attr ty) (arg: TypedSSAVal ty): Op attr ty :=
  match o with
  | Op.mk name res args regions attrs =>
    Op.mk name res (args ++ [arg])  regions attrs

def Op.addResult (o: Op attr ty) (new_res: TypedSSAVal ty): Op attr ty :=
 match o with
 | Op.mk name res args  regions attrs =>
    Op.mk name (res ++ [new_res]) args  regions attrs

def Op.appendRegion (o: Op attr ty) (r: Region attr ty): Op attr ty :=
  match o with
  | Op.mk name res args regions attrs =>
      Op.mk name res args (regions ++ [r]) attrs


-- | Note: AttrEntry can be given as String × AttrValue
def AttrDict.add [Code attr] (attrs: AttrDict attr) (entry: AttrEntry attr): AttrDict attr :=
    (entry :: attrs)

def AttrDict.find [Code attr] (attrs: AttrDict attr) (name: String): Option (AttrValue attr) :=
  match attrs with
  | AttrDict.mk entries =>
      match entries.find? (fun entry => entry.key == name) with
      | some v => v.value
      | none => none

def AttrDict.find_nat [Code attr] (attrs: AttrDict attr)
  (name: String): Option Nat :=
  match attrs.find name with
  | .some (AttrValue.nat i) =>  .some i
  | _ => .none

def AttrDict.find_int [Code attr] (attrs: AttrDict attr)
  (name: String): Option Int :=
  match attrs.find name with
  | .some (AttrValue.int i) => .some i
  | _ => .none

def AttrDict.find_int' [Code attr] (attrs: AttrDict attr) (name: String): Option Int :=
  match attrs.find name with
  | .some (AttrValue.int i) =>  .some i
  | _ => .none

@[simp] theorem AttrDict.find_none [Code attr]:
    AttrDict.find (attr := attr) (AttrDict.mk []) name = none := by
  simp [AttrDict.find, List.find?]

@[simp] theorem AttrDict.find_next [Code attr] (v: AttrValue attr)
  (l: List (AttrEntry attr)):
    AttrDict.find (AttrDict.mk (AttrEntry.mk n v :: l)) n' =
    if n == n' then some v else AttrDict.find (AttrDict.mk l) n' := by
  cases H: n == n' <;>
  simp [AttrDict.find, List.find?, AttrEntry.key, AttrEntry.value, H]

def AttrDict.addString (attrs: AttrDict δ) (k: String) (v: String): AttrDict δ :=
    AttrEntry.mk k (v: AttrValue δ) :: attrs


def Op.addAttr [Code attr] (o: Op attr ty) (k: String) (v: AttrValue attr): Op attr ty :=
 match o with
 | Op.mk name res args regions attrs =>
    Op.mk name res args regions (attrs.add (k, v))

def Region.empty (name: String): Region attr ty := Region.mk name [] []
def Region.appendOp (bb: Region attr ty) (op: Op attr ty): Region attr ty :=
  match bb with
  | Region.mk name args bbs => Region.mk name args (bbs ++ [op])

def Region.appendOps (bb: Region attr ty) (ops: List (Op attr ty)): Region attr ty :=
  match bb with
  | Region.mk name args bbs => Region.mk name args (bbs ++ ops)


instance [Code attr] [Code ty]: Pretty (Op attr ty) where
  doc := op_to_doc

instance [Code attr] [Code ty]: Pretty (Region attr ty) where
  doc := rgn_to_doc

instance [Pretty a] : ToString a where
  toString (v: a) := layout80col (doc v)

instance [Code attr] [Code ty]: ToFormat (Op attr ty) where
    format (x: Op attr ty) := layout80col (doc x)


instance : Inhabited (MLIRType ty) where
  default := MLIRType.undefined "INHABITANT"

instance : Inhabited (AttrValue attr) where
  default := AttrValue.str "INHABITANT"

instance : Inhabited (Op attr ty) where
  default := Op.empty "INHABITANT"

instance : Inhabited (Region attr ty) where
  default := Region.empty "INHABITANT"

instance  [Code attr] [Code ty]: Pretty (Module attr ty) where
  doc (m: Module attr ty) :=
    match m with
    | Module.mk fs attrs =>
      Doc.VGroup (attrs.map doc ++ fs.map doc)

def Region.fromOps (os: List (Op attr ty)) (name: String := "entry"): Region attr ty :=
  Region.mk name [] os

def Region.setArgs (bb: Region attr ty) (args: List (TypedSSAVal ty)) : Region attr ty :=
match bb with
  | (Region.mk name _ ops) => (Region.mk name args ops)


end MLIR.AST
