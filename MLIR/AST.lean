import MLIR.Doc
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

deriving instance DecidableEq for SSAVal

#check SSAVal


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

mutual

inductive MemrefLayoutSpec : Type where
| stride: (offset: Dimension) -> (stride: List Dimension) -> MemrefLayoutSpec
| attr: AttrVal -> MemrefLayoutSpec

inductive MLIRTy : Type where
| fn : MLIRTy -> MLIRTy -> MLIRTy
| int : Int -> MLIRTy
| float: Int -> MLIRTy
| index:  MLIRTy
| tuple : List MLIRTy -> MLIRTy
| vector: (fixed: (List Int)) -> (scaled: (List Int)) -> MLIRTy -> MLIRTy
| tensorRanked: (dims: List Dimension) -> (ty: MLIRTy) -> (layout: Option AttrVal) -> MLIRTy
| tensorUnranked: MLIRTy -> MLIRTy
| memrefRanked: (dims: List Dimension) -> (t: MLIRTy) ->
  (layout: Option MemrefLayoutSpec) -> (memspace: Option AttrVal) -> MLIRTy
| memrefUnranked:  (t: MLIRTy) ->  (memspace: Option AttrVal) -> MLIRTy
-- | TODO: combine user, userPretty, userGeneric
| userPretty: (dialect: String) -> (name: String) -> (contents: String) -> MLIRTy
| userGeneric: (dialect: String) -> (contents: Option String) -> MLIRTy

-- | TODO: factor Symbol out from AttrVal
inductive AttrVal : Type where
| symbol: String -> AttrVal -- symbol ref attr
| str : String -> AttrVal
| int : Int -> MLIRTy -> AttrVal
| bool : Bool -> AttrVal
| float : Float -> MLIRTy -> AttrVal
| type :MLIRTy -> AttrVal
| dense: TensorElem -> MLIRTy -> AttrVal -- dense<10> : vector<i32>
| affine: AffineMap -> AttrVal
| list: List AttrVal -> AttrVal
-- | guaranteee: both components will be AttrVal.Symbol.
-- | TODO: factor symbols out.
| nestedsymbol: AttrVal -> AttrVal -> AttrVal
| alias: String -> AttrVal
| dict: AttrDict -> AttrVal
| opaque: (dialect: String) -> (value: String) -> AttrVal
| opaqueElements: (dialect: String) -> (value: String) -> (type: MLIRTy) -> AttrVal
-- | userPrettyDialectItem: (dialect: String) -> (name: String) -> (contents: String) -> MLIRTy
| unit: AttrVal

-- https://mlir.llvm.org/docs/LangRef/#attributes
-- | TODO: add support for mutually inductive records / structures
inductive AttrEntry : Type where
  | mk: (key: String)
      -> (value: AttrVal)
      -> AttrEntry

inductive AttrDict : Type :=
| mk: List AttrEntry -> AttrDict
end


mutual
-- | TODO: make this `record` when mutual records are allowed?
-- | TODO: make these arguments optional?
inductive Op : Type where
 | mk: (name: String)
      -> (args: List SSAVal)
      -> (bbs: List BBName)
      -> (regions: List Region)
      -> (attrs: AttrDict)
      -> (ty: MLIRTy)
      -> Op

inductive BasicBlockStmt : Type where
| StmtAssign : SSAVal -> (ix: Option Int) -> Op ->BasicBlockStmt
| StmtOp : Op -> BasicBlockStmt


inductive BasicBlock: Type where
| mk: (name: String)
      -> (args: List (SSAVal × MLIRTy))
      -> (ops: List BasicBlockStmt) -> BasicBlock

inductive Region: Type where
| mk: (bbs: List BasicBlock) -> Region

end



def Op.name: Op -> String
| Op.mk name args bbs regions attrs ty => name


def Op.args: Op -> List SSAVal
| Op.mk name args bbs regions attrs ty => args

def Op.bbs: Op -> List BBName
| Op.mk name args bbs regions attrs ty => bbs


def Op.regions: Op -> List Region
| Op.mk name args bbs regions attrs ty => regions

def Op.attrs: Op ->  AttrDict
| Op.mk name args bbs regions attrs ty => attrs

def Op.ty: Op ->  MLIRTy
| Op.mk name args bbs regions attrs ty => ty

def Region.bbs (r: Region): List BasicBlock :=
  match r with
  | (Region.mk bbs) => bbs


inductive AttrDefn where
| mk: (name: String) -> (val: AttrVal) -> AttrDefn

def BasicBlockStmt.op: BasicBlockStmt->  Op
| BasicBlockStmt.StmtAssign val ix op => op
| BasicBlockStmt.StmtOp op => op

def BasicBlock.name (bb: BasicBlock): BBName :=
  match bb with
  | BasicBlock.mk name args stmts => BBName.mk name


def BasicBlock.stmts (bb: BasicBlock): List BasicBlockStmt :=
  match bb with
  | BasicBlock.mk name args stmts => stmts

def Region.getBasicBlock (r: Region) (name: BBName): Option BasicBlock :=
  r.bbs.find? (fun bb => bb.name == name)



-- | TODO: this seems like a weird exception. Is this really true?
inductive Module where
| mk: (functions: List Op)
      -> (attrs: List AttrDefn)
      ->  Module


def MLIRTy.beq (t1 t2: MLIRTy): Bool :=
  match t1, t2 with
  | MLIRTy.fn a1 b1, MLIRTy.fn a2 b2 =>
      beq a1 a2 && beq b1 b2
  | MLIRTy.int n1, MLIRTy.int n2 =>
      n1 == n2
  | MLIRTy.float n1, MLIRTy.float n2 =>
      n1 == n2
  | MLIRTy.index, MLIRTy.index =>
      true
  | MLIRTy.tuple [], MLIRTy.tuple [] =>
      true
  | MLIRTy.tuple (t1::l1), MLIRTy.tuple (t2::l2) =>
      beq t1 t2 && beq (MLIRTy.tuple l1) (MLIRTy.tuple l2)
  | MLIRTy.vector fixed1 scaled1 t1, MLIRTy.vector fixed2 scaled2 t2 =>
      fixed1 = fixed2 && scaled1 = scaled2 && beq t1 t2
  | MLIRTy.tensorRanked l1 t1 a1, MLIRTy.tensorRanked l2 t2 a2 =>
      l1 == l2 && beq t1 t2
  | MLIRTy.tensorUnranked t1, MLIRTy.tensorUnranked t2 =>
      beq t1 t2
  | MLIRTy.memrefRanked dims1 t1 _ _, MLIRTy.memrefRanked dims2 t2 _ _ =>
      -- | TODO: MLIRTy.beq: Also compare memref settings?
      dims1 == dims2 && beq t1 t2
  | MLIRTy.memrefUnranked t1 _, MLIRTy.memrefUnranked t2 _ =>
      beq t1 t2
  | MLIRTy.userGeneric n1 k1, MLIRTy.userGeneric n2 k2 =>
      n1 == n2
  | _, _ =>
      false

def MLIRTy.decEq (t1 t2: MLIRTy): Decidable (Eq t1 t2) :=
  if MLIRTy.beq t1 t2 then isTrue sorry else isFalse sorry

instance: DecidableEq MLIRTy :=
  MLIRTy.decEq


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
mutual

partial def docMemrefLayoutSpec(spec: MemrefLayoutSpec) : Doc :=
match spec with
| MemrefLayoutSpec.stride offset strides => [doc| "offset:" offset ", strides: " "[" (strides),* "]"]
|  MemrefLayoutSpec.attr v => docAttrVal v


partial def docMlirTy(ty: MLIRTy) : Doc :=
    let rec  go (ty: MLIRTy) :=
    match ty with
    | MLIRTy.userGeneric name contents? => 
      match contents? with 
      | some c => [doc| "!"name"<"c">"] -- TODO: does this print "" around strings?
      | none => [doc| "!"name]
    | MLIRTy.userPretty dialect name contents => [doc| "!" dialect "." name "<" contents ">"]
    | MLIRTy.int k => [doc| "i"k]
    | MLIRTy.float k => [doc| "f"k]
    | MLIRTy.index => [doc| "index"]
    | MLIRTy.tuple ts => [doc| "(" (ts.map go),* ")" ]
    | MLIRTy.fn dom codom => (go dom) ++ " -> " ++ (go codom)
    | MLIRTy.vector fixed scaled ty =>
      let docFixed := match fixed with
        | [] => ""
        | _ => (intercalate_doc fixed "×") ++ "×"
      let docScaling := match scaled with
        | [] => ""
        | _ => (intercalate_doc fixed "×") ++ "×"
      [doc| "vector<" (docFixed) (docScaling) (go ty) ">"]
    | MLIRTy.memrefRanked dims ty layout? memspace? =>
      let docLayout := match layout? with | some x => [doc| "," (docMemrefLayoutSpec x)] | none => ""
      let docMemspace := match memspace? with | some x => [doc| "," (docAttrVal x)] | none => ""
      [doc| "memref<" (intercalate_doc dims "x") "x" (go ty) (docLayout)  (docMemspace) ">"]
    | MLIRTy.memrefUnranked ty memspace? =>
      let docMemspace := match memspace? with | some x => [doc| "," (docAttrVal x)] | none => ""
      [doc| "memref<" "*x" (go ty) (docMemspace) ">"]
    | MLIRTy.tensorRanked dims ty none => "tensor<" ++ (intercalate_doc dims "x") ++ "x" ++ go ty ++ ">"
    | MLIRTy.tensorRanked dims ty (some layout) => "tensor<" ++ (intercalate_doc dims "x") ++ "x" ++ go ty ++ "," ++ docAttrVal layout ++ ">"
    | MLIRTy.tensorUnranked ty => "tensor<" ++ "*x" ++ go ty ++ ">"
    go ty

partial def docAttrVal (v: AttrVal) :=
   match v with
   | AttrVal.symbol s => "@" ++ doc_surround_dbl_quot s
   | AttrVal.nestedsymbol s t => (docAttrVal s) ++ "::" ++ (docAttrVal t)
   | AttrVal.str str => doc_surround_dbl_quot str
   | AttrVal.type ty => docMlirTy ty
   | AttrVal.int i ty => doc i ++ " : " ++ docMlirTy ty
   | AttrVal.bool b => if b then "true" else "false"
   | AttrVal.float f ty => doc f ++ " : " ++ docMlirTy ty
   | AttrVal.dense elem ty => "dense<" ++ doc elem ++ ">" ++ ":" ++ docMlirTy ty
   | AttrVal.affine aff => "affine_map<" ++ doc aff ++ ">"
   | AttrVal.list xs => "[" ++ Doc.Nest (vintercalate_doc (xs.map docAttrVal) ", ") ++ "]"
   | AttrVal.alias a => "#" ++ a
   | AttrVal.dict d => docAttrDict d
   | AttrVal.opaque dialect val => [doc| "#" (dialect) "<"  (val) ">"]
   | AttrVal.opaqueElements dialect val ty => [doc| "#opaque<" (dialect) ","  (val) ">" ":" (docMlirTy ty)]
   | AttrVal.unit => "()"

partial def docAttrEntry (a: AttrEntry) :=
    match a with
    | AttrEntry.mk k v => k ++ " = " ++ (docAttrVal v)


partial def docAttrDict (v: AttrDict) :=
   match v with
   | AttrDict.mk attrs =>
        if List.isEmpty attrs
        then Doc.Text ""
        else "{" ++ Doc.Nest (vintercalate_doc (attrs.map docAttrEntry)  ", ")  ++ "}"
end

partial instance : Pretty MemrefLayoutSpec where
 doc  := docMemrefLayoutSpec

partial instance : Pretty MLIRTy where
 doc := docMlirTy

partial instance : Pretty AttrVal where
 doc (v: AttrVal) := docAttrVal v

instance : Pretty AttrEntry where
  doc (a: AttrEntry) := docAttrEntry a


 instance : Pretty AttrDict where
   doc v := docAttrDict v

instance : Pretty AttrDefn where
  doc (v: AttrDefn) :=
  match v with
  | AttrDefn.mk name val => "#" ++ name ++ " := " ++ (doc val)


instance: Coe String SSAVal where
  coe (s: String) := SSAVal.SSAVal s

instance : Coe Int TensorElem where
  coe (i: Int) := TensorElem.int i

instance : Coe  (List Int) TensorElem where
  coe (xs: List Int) := TensorElem.nested (xs.map TensorElem.int)

instance : Coe String AttrVal where
  coe (s: String) := AttrVal.str s

instance : Coe Int AttrVal where
  coe (i: Int) := AttrVal.int i (MLIRTy.int 64)

instance : Coe MLIRTy AttrVal where
  coe (t: MLIRTy) := AttrVal.type t


-- | create a dense vector with values 'xs' and type vector<len(xs)xity>
def AttrVal.dense_vector (xs: List Int) (ity: MLIRTy := MLIRTy.int 32): AttrVal :=
  let fixedShape := [Int.ofNat xs.length]
  let scaledShape := []
  let vty := MLIRTy.vector fixedShape scaledShape ity
  AttrVal.dense xs vty

instance : Coe (String × AttrVal) AttrEntry where
  coe (v: String × AttrVal) := AttrEntry.mk v.fst v.snd

instance : Coe (String × MLIRTy) AttrEntry where
  coe (v: String × MLIRTy) := AttrEntry.mk v.fst (AttrVal.type v.snd)

instance : Coe  AttrEntry (String × AttrVal) where
  coe (v: AttrEntry) :=
  match v with
  | AttrEntry.mk key val => (key, val)


instance : Coe (List AttrEntry) AttrDict where
  coe (v: List AttrEntry) := AttrDict.mk v

 instance : Coe AttrDict (List AttrEntry) where
  coe (v: AttrDict) := match v with | AttrDict.mk as => as


instance : Coe (BasicBlock) Region where
  coe (bb: BasicBlock) := Region.mk [bb]

instance : Coe (List BasicBlock) Region where
  coe (bbs: List BasicBlock) := Region.mk bbs

instance : Coe  Region (List BasicBlock) where
  coe (rgn: Region) := match rgn with | Region.mk bbs => bbs

instance : Pretty SSAVal where
   doc (val: SSAVal) :=
     match val with
     | SSAVal.SSAVal name => Doc.Text ("%" ++ name)


instance : ToFormat SSAVal where
    format (x: SSAVal) := layout80col (doc x)


-- | TODO: allow mutual definition of typeclass instances. This code
-- | would be so much nicer if I could pretend that these had real instances.
mutual

partial instance : Pretty Op where
  doc := op_to_doc

partial instance : Pretty BasicBlock where
  doc := bb_to_doc

partial instance : Pretty Region where
  doc := rgn_to_doc

partial def op_to_doc (op: Op): Doc :=
    match op with
    | (Op.mk name args bbs rgns attrs ty) =>
        /- v3: macros + if stuff-/
        [doc|
          "\"" name "\""
          "(" (args),* ")"
          (ifdoc bbs.isEmpty then "" else  "[" (bbs),* "]")
          (ifdoc rgns.isEmpty then "" else  "(" (nest rgns.map rgn_to_doc);* ")")
          attrs
          ":"
          ty
        ]

        /- v2: macros, but no if stuff
        [doc|
          "\"" name "\""
          "(" (args),* ")"
          (if bbs.isEmpty then [doc| ""] else [doc| "[" (bbs),* "]"])
          (if rgns.isEmpty then [doc| ""] else[doc| "(" (nest rgns.map rgn_to_doc);* ")"])
          attrs
          ":"
          ty
        ]
        -/

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

        doc_name ++ doc_args ++  doc_bbs ++ doc_rgns ++ doc attrs ++ " : " ++ doc ty
        -/
-- partial def bb_stmt_to_doc (stmt: BasicBlockStmt): Doc :=
--  match stmt with
--  | BasicBlockStmt.StmtAssign lhs rhs => (doc lhs) ++ " = " ++ (op_to_doc rhs)
--  | BasicBlockStmt.StmtOp rhs => (op_to_doc rhs)

partial def bb_stmt_to_doc (stmt: BasicBlockStmt): Doc :=
  match stmt with
  | BasicBlockStmt.StmtAssign lhs none rhs =>
      [doc| lhs "="  (op_to_doc rhs) ]
  | BasicBlockStmt.StmtAssign lhs (some ix) rhs =>
      [doc| lhs ":" ix "="  (op_to_doc rhs) ]
  | BasicBlockStmt.StmtOp rhs => (op_to_doc rhs)


-- | TODO: fix the dugly syntax
partial def bb_to_doc(bb: BasicBlock): Doc :=
  let doc_arg (arg: SSAVal × MLIRTy) :=
        match arg with | (ssaval, ty) => [doc| ssaval ":" ty]
  match bb with
  | (BasicBlock.mk name args stmts) =>
    [doc|
      {
        (ifdoc args.isEmpty
         then  "^" name ":"
         else  "^" name "(" (args.map $ fun (v, t) => [doc| v ":" t]),* ")" ":");
        (nest stmts.map bb_stmt_to_doc);* ;
      }
    ]

partial def rgn_to_doc(rgn: Region): Doc :=
  match rgn with
  | (Region.mk bbs) => [doc| { "{"; (nest (bbs.map bb_to_doc);* ); "}"; }]


end

def AttrEntry.key (a: AttrEntry): String :=
match a with
| AttrEntry.mk k v => k

def AttrEntry.value (a: AttrEntry): AttrVal :=
match a with
| AttrEntry.mk k v => v


def MLIRTy.unit : MLIRTy := MLIRTy.tuple []
def AttrDict.empty : AttrDict := AttrDict.mk []

def Op.empty (name: String) : Op :=
  Op.mk name [] [] [] AttrDict.empty (MLIRTy.fn MLIRTy.unit MLIRTy.unit)
-- | TODO: needs to happen in a monad to ensure that ty has the right type!
def Op.addArg (o: Op) (a: SSAVal) (t: MLIRTy): Op :=
  match o with
  | Op.mk name args bbs regions attrs ty =>
    let ty' := match ty with
               | MLIRTy.fn (MLIRTy.tuple ins) outs =>
                           MLIRTy.fn (MLIRTy.tuple $ ins ++ [t]) outs
               | _ => MLIRTy.fn (MLIRTy.tuple [t]) (MLIRTy.unit)
    Op.mk name (args ++ [a]) bbs regions attrs ty'

def Op.addResult (o: Op) (t: MLIRTy): Op :=
 match o with
 | Op.mk name args bbs regions attrs ty =>
    let ty' := match ty with
               | MLIRTy.fn ins (MLIRTy.tuple outs) =>
                           MLIRTy.fn ins (MLIRTy.tuple $ outs ++ [t])
               | _ => MLIRTy.fn (MLIRTy.tuple []) (MLIRTy.tuple [t])
    Op.mk name args bbs regions attrs ty'

def Op.appendRegion (o: Op) (r: Region): Op :=
  match o with
  | Op.mk name args bbs regions attrs ty =>
      Op.mk name args bbs (regions ++ [r]) attrs ty


-- | Note: AttrEntry can be given as String × AttrVal
def AttrDict.add (attrs: AttrDict) (entry: AttrEntry): AttrDict :=
    Coe.coe $ (entry :: Coe.coe attrs)

def AttrDict.find (attrs: AttrDict) (name: String): Option AttrVal :=
  match attrs with
  | AttrDict.mk entries =>
      match entries.find? (fun entry => entry.key == name) with
      | some v => v.value
      | none => none

def AttrDict.addString (attrs: AttrDict) (k: String) (v: String): AttrDict :=
    Coe.coe $ ((AttrEntry.mk k (Coe.coe v)) :: Coe.coe attrs)

def AttrDict.addType (attrs: AttrDict) (k: String) (v: MLIRTy): AttrDict :=
    Coe.coe $ ((AttrEntry.mk k (Coe.coe v)) :: Coe.coe attrs)


-- | Note: AttrEntry can be given as String × AttrVal
def Op.addAttr (o: Op) (k: String) (v: AttrVal): Op :=
 match o with
 | Op.mk name args bbs regions attrs ty =>
    Op.mk name args bbs regions (attrs.add (k, v)) ty

def BasicBlock.empty (name: String): BasicBlock := BasicBlock.mk name [] []
def BasicBlock.appendStmt (bb: BasicBlock) (stmt: BasicBlockStmt): BasicBlock :=
  match bb with
  | BasicBlock.mk name args bbs => BasicBlock.mk name args (bbs ++ [stmt])

def BasicBlock.appendStmts (bb: BasicBlock) (stmts: List BasicBlockStmt): BasicBlock :=
  match bb with
  | BasicBlock.mk name args bbs => BasicBlock.mk name args (bbs ++ stmts)

def Region.empty: Region := Region.mk []

def Region.appendBasicBlock (r: Region) (bb: BasicBlock) : Region :=
  Coe.coe (Coe.coe r ++ [bb])

instance : Pretty Op where
  doc := op_to_doc

instance : Pretty BasicBlockStmt where
  doc := bb_stmt_to_doc

instance : Pretty BasicBlock where
  doc := bb_to_doc

instance : Pretty Region where
  doc := rgn_to_doc

instance [Pretty a] : ToString a where
  toString (v: a) := layout80col (doc v)

instance : ToFormat Op where
    format (x: Op) := layout80col (doc x)


instance : Inhabited Op where
  default := Op.empty "INHABITANT"

instance : Inhabited BasicBlock where
  default := BasicBlock.empty "INHABITANT"

instance : Inhabited Region where
  default := Region.empty

instance : Pretty Module where
  doc (m: Module) :=
    match m with
    | Module.mk fs attrs =>
      Doc.VGroup (attrs.map doc ++ fs.map doc)

instance : Coe Op BasicBlockStmt where
   coe := BasicBlockStmt.StmtOp

def Region.fromBlock (bb: BasicBlock): Region := Region.mk [bb]
def BasicBlock.fromOps (os: List Op) (name: String := "entry") :=
  BasicBlock.mk name [] (os.map BasicBlockStmt.StmtOp)

def BasicBlock.setArgs (bb: BasicBlock) (args: List (SSAVal × MLIRTy)) : Region :=
match bb with
  | (BasicBlock.mk name _ stmts) => (BasicBlock.mk name args stmts)

def Region.fromOps (os: List Op): Region := Region.mk [BasicBlock.fromOps os]

-- | return the only region in the block
def Op.singletonRegion (o: Op): Region :=
  match o.regions with
  | (r :: []) => r
  | _ => panic! "expected op with single region: " ++ (doc o)

def Op.mutateSingletonRegion (o: Op) (f: Region -> Region): Op :=
 match o with
 | Op.mk name args bbs [r] attrs ty => Op.mk name args bbs [f r] attrs ty
 | _ => panic! "expected op with single region: " ++ (doc o)


mutual
-- | TODO: how the fuck do we run this lens?!
inductive ValLens: Type _ -> Type _ where
| id: ValLens (ULift SSAVal)
| op: (opKind: String) -> (lens: OpLens (o: Type u)) -> ValLens o

inductive OpLens: Type _ -> Type _ where
| region: Nat -> RegionLens (o: Type u) -> OpLens o
| id: OpLens (ULift Op)
| arg: Nat -> ValLens (o: Type u) -> OpLens o

inductive RegionLens: Type _ -> Type _ where
| block: Nat -> BasicBlockLens (o: Type u) -> RegionLens o
| id: RegionLens (ULift Region)

inductive BasicBlockLens: Type _ -> Type _ where
| op: Nat -> OpLens (o: Type u) -> BasicBlockLens o
| id: BasicBlockLens (ULift BasicBlock)
end


-- | defunctionalized lens where `s` is lensed by `l t` to produce a `t`
class Lensed (s: Type _) (l: Type _ -> Type _) where
  lensId: l (ULift s) -- create the identity lens for this source
  -- view: s -> l t -> t -- view using the lens at the target
  update: [Applicative f] -> l t -> (t -> f t) -> (s -> f s)


-- | ignore the x
structure PointedConst (t: Type) (v: t) (x: Type) where
  (val: t)


instance : Coe (PointedConst t v x) (PointedConst t v y) where
  coe a := { val := a.val }

instance: Functor (PointedConst t v) where
  map f p := p

instance: Pure (PointedConst t v) where
  pure x := PointedConst.mk v

instance: SeqLeft (PointedConst t v) where
  seqLeft pc _ := pc

instance: SeqRight (PointedConst t v) where
  seqRight pc _ := pc

instance: Seq (PointedConst t v) where
  seq f pc := pc ()

instance : Applicative (PointedConst t v) where
  map      := fun x y => Seq.seq (pure x) fun _ => y
  seqLeft  := fun a b => Seq.seq (Functor.map (Function.const _) a) b
  seqRight := fun a b => Seq.seq (Functor.map (Function.const _ id) a) b


def Lensed.get [Lensed s l] (lens: l t) (sval: s) (default_t: t): t :=
  (Lensed.update lens (fun t => @PointedConst.mk _ default_t _ t) sval).val

def Lensed.set [Lensed s l] (lens: l t) (sval: s) (tnew: t): s :=
  Lensed.update (f := Id) lens (fun t => tnew) sval

def Lensed.map [Lensed s l] (lens: l t) (sval: s) (tfun: t -> t): s :=
  Lensed.update (f := Id) lens tfun sval

def Lensed.mapM [Lensed s l]  [Monad m] (lens: l t) (sval: s) (tfun: t -> m t): m s :=
  Lensed.update lens tfun sval

-- | TODO: for now, when lens fails, we just return.
mutual
-- | how can this lens ever be run? Very interesting...
def vallens_update {f: Type -> Type} {t: Type} [Applicative f] (lens: ValLens t) (transform: t -> f t) (src: SSAVal) : f SSAVal :=
    match lens with
    | ValLens.id => Functor.map ULift.down $ transform (ULift.up src)
    | ValLens.op kind oplens => Pure.pure src -- TODO: how do we encode this?

def oplens_update {f: Type -> Type} {t: Type} [Applicative f] (lens: OpLens t) (transform: t -> f t) (src: Op) : f Op :=
    match lens with
    | OpLens.id => Functor.map ULift.down $ transform (ULift.up src)
    | OpLens.arg ix vlens =>
        match src with
        | Op.mk name args bbs regions attrs ty =>
        match args.get? ix with
        | none => Pure.pure src
        | some v => Functor.map (fun v => Op.mk name (args.set ix v) bbs regions attrs ty)
                               (vallens_update vlens transform v)
    | OpLens.region ix rlens =>
      match src with
      | Op.mk name args bbs  regions attrs ty =>
      match regions.get? ix with
      | none => Pure.pure src
      | some r =>  Functor.map (fun r => Op.mk name args bbs (regions.set ix r) attrs ty)
                               (regionlens_update rlens transform r)

def regionlens_update {f: Type -> Type} {t: Type} [Applicative f] (lens: RegionLens t) (transform: t -> f t) (src: Region) : f Region :=
    match lens with
    | RegionLens.id => Functor.map ULift.down $ transform (ULift.up src)
    | RegionLens.block ix bblens =>
      match src with
      | Region.mk bbs =>
        match bbs.get? ix with
        | none => Pure.pure src
        | some bb =>  Functor.map (fun bb => Region.mk (bbs.set ix bb)) (blocklens_update bblens transform bb)


def blocklens_update {f: Type -> Type} {t: Type}[Applicative f] (lens: BasicBlockLens t) (transform: t -> f t) (src: BasicBlock) : f BasicBlock :=
    match lens with
    | BasicBlockLens.id => Functor.map ULift.down $ transform (ULift.up src)
    | BasicBlockLens.op ix oplens =>
      match src with
      | BasicBlock.mk name args ops =>
        match ops.get? ix with
        | none => Pure.pure src
        | some stmt =>
            let stmt := match stmt with
            | BasicBlockStmt.StmtAssign lhs ix op => (BasicBlockStmt.StmtAssign lhs ix) <$> (oplens_update oplens transform op)
            | BasicBlockStmt.StmtOp op => BasicBlockStmt.StmtOp <$> (oplens_update oplens transform op)
            (fun stmt => BasicBlock.mk name args (ops.set ix stmt)) <$> stmt
end

instance : Lensed Op OpLens where
  lensId := OpLens.id
  update := oplens_update

instance : Lensed Region RegionLens where
  lensId := RegionLens.id
  update := regionlens_update

instance : Lensed BasicBlock BasicBlockLens where
  lensId := BasicBlockLens.id
  update := blocklens_update

def Region.singletonBlock (r: Region): BasicBlock :=
  match r.bbs with
  | (bb :: []) => bb
  | _ => panic! "expected region with single bb: " ++ (doc r)

-- | Ensure that region has an entry block.
def Region.ensureEntryBlock (r: Region): Region :=
match r with
| (Region.mk bbs) =>
  match bbs with
  | []  => BasicBlock.empty "entry"
  | _ => r


-- | replace entry block arguments.
def Region.setEntryBlockArgs (r: Region) (args: List (SSAVal × MLIRTy)) : Region :=
match r with
| (Region.mk bbs) =>
  match bbs with
  | []  => r
  | ((BasicBlock.mk name _ stmts) :: bbs) => Region.mk $ (BasicBlock.mk name args stmts) :: bbs

end MLIR.AST
