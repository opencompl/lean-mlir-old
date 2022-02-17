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

inductive AffineTuple 
| mk: List AffineExpr -> AffineTuple

instance : Pretty AffineTuple where
  doc t := match t with
  | AffineTuple.mk es => "(" ++ intercalate_doc es "," ++ ")" 
 
inductive AffineMap
| mk: AffineTuple -> AffineTuple -> AffineMap

 instance : Pretty AffineMap where
  doc t := match t with
  | AffineMap.mk xs ys => doc xs ++ " -> " ++ doc ys
 
 

-- EMBEDDING
-- ==========

inductive BBName
| mk: String -> BBName

instance : Pretty BBName where
  doc name := match name with 
              | BBName.mk s => "^" ++ doc s


inductive Dimension
| Known: Int -> Dimension
| Unknown: Dimension

mutual
inductive MLIRTy : Type where
| fn : MLIRTy -> MLIRTy -> MLIRTy
| int : Int -> MLIRTy
| float: Int -> MLIRTy
| tuple : List MLIRTy -> MLIRTy
| vector: Int -> MLIRTy -> MLIRTy
| tensor: List Dimension -> MLIRTy -> MLIRTy
| user: String -> MLIRTy -- user defined type

inductive SSAVal : Type where
  | SSAVal : String -> SSAVal

inductive AttrVal : Type where
| symbol: String -> AttrVal -- symbol ref attr
| str : String -> AttrVal
| int : Int -> MLIRTy -> AttrVal
| type :MLIRTy -> AttrVal
| dense: Int -> MLIRTy -> AttrVal -- dense<10> : vector<i32>
| affine: AffineMap -> AttrVal
| list: List AttrVal -> AttrVal

-- https://mlir.llvm.org/docs/LangRef/#attributes
inductive AttrEntry : Type where
  | mk: (key: String) 
      -> (value: AttrVal)
      -> AttrEntry

inductive AttrDict : Type := 
| mk: List AttrEntry -> AttrDict


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
| StmtAssign : SSAVal -> Op -> BasicBlockStmt
| StmtOp : Op -> BasicBlockStmt


inductive BasicBlock: Type where
| mk: (name: String) -> (args: List (SSAVal × MLIRTy)) -> (ops: List BasicBlockStmt) -> BasicBlock

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



-- | TODO: this seems like a weird exception. Is this really true?
inductive Module where
| mk: (functions: List Op) 
      -> (attrs: List AttrDefn) 
      ->  Module

instance : Pretty Dimension where
  doc dim := 
  match dim with
  | Dimension.Unknown => "?"
  | Dimension.Known i => doc i

partial instance :  Pretty MLIRTy where
  doc (ty: MLIRTy) :=
    let rec  go (ty: MLIRTy) :=  
    match ty with
    | MLIRTy.user k => "!" ++ k
    | MLIRTy.int k => "i" ++ doc k
    | MLIRTy.float k => "f" ++ doc k
    | MLIRTy.tuple ts => "(" ++ (intercalate_doc (ts.map go) (doc ", ") ) ++ ")"
    | MLIRTy.fn dom codom => (go dom) ++ " -> " ++ (go codom)
    | MLIRTy.vector sz ty => "vector<" ++ toString sz ++ "x" ++ go ty ++ ">"
    | MLIRTy.tensor dims ty => "tensor<" ++ (intercalate_doc dims "x") ++ "x" ++ go ty ++ ">"
    go ty



partial instance : Pretty AttrVal where
 doc (v: AttrVal) := 
  let rec go (v: AttrVal) :=
   match v with
   | AttrVal.symbol s => "@" ++ doc_surround_dbl_quot s
   | AttrVal.str str => doc_surround_dbl_quot str 
   | AttrVal.type ty => doc ty
   | AttrVal.int i ty => doc i ++ " : " ++ doc ty
   | AttrVal.dense i ty => "dense<" ++ doc i ++ ">" ++ ":" ++ doc ty
   | AttrVal.affine aff => "affine_map<" ++ doc aff ++ ">" 
   | AttrVal.list xs => "[" ++ Doc.Nest (vintercalate_doc (xs.map go) ", ") ++ "]"
  go v

instance : Pretty AttrEntry where
  doc (a: AttrEntry) := 
    match a with
    | AttrEntry.mk k v => k ++ " = " ++ (doc v)

instance : Pretty AttrDefn where
  doc (v: AttrDefn) := 
  match v with
  | AttrDefn.mk name val => "#" ++ name ++ " := " ++ (doc val)
 

 instance : Pretty AttrDict where
   doc v := match v with
   | AttrDict.mk attrs => 
        if List.isEmpty attrs
        then Doc.Text ""
        else "{" ++ Doc.Nest (vintercalate_doc attrs ", ")  ++ "}" 

instance : Coe (String × AttrVal) AttrEntry where 
  coe (v: String × AttrVal) := AttrEntry.mk v.fst v.snd

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





-- | TODO: add a typeclass `Pretty` for things that can be converted to `Doc`.
mutual
partial def op_to_doc (op: Op): Doc := 
    match op with
    | (Op.mk name args bbs rgns attrs ty) => 
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

partial def bb_stmt_to_doc (stmt: BasicBlockStmt): Doc :=
  match stmt with
  | BasicBlockStmt.StmtAssign lhs rhs => (doc lhs) ++ " = " ++ (op_to_doc rhs)
  | BasicBlockStmt.StmtOp rhs => (op_to_doc rhs)

partial def bb_to_doc(bb: BasicBlock): Doc :=
  match bb with
  | (BasicBlock.mk name args stmts) => 
     let doc_arg (arg: SSAVal × MLIRTy) := 
        match arg with
        | (ssaval, ty) => doc ssaval ++ ":" ++ doc ty
     let bbargs := 
        if args.isEmpty then Doc.Text ""
        else "(" ++ (intercalate_doc (args.map doc_arg) ", ") ++ ")"
     let bbname := "^" ++ name ++ bbargs ++ ":"
     let bbbody := Doc.Nest (Doc.VGroup (stmts.map bb_stmt_to_doc))
     Doc.VGroup [bbname, bbbody]

partial def rgn_to_doc(rgn: Region): Doc :=
  match rgn with
  | (Region.mk bbs) => "{" ++ Doc.VGroup [nest_vgroup (bbs.map bb_to_doc), "}"]
 
end

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
    coe $ (entry :: coe attrs)

-- | Note: AttrEntry can be given as String × AttrVal
def Op.addAttr (o: Op) (k: String) (entry: AttrEntry): Op :=
 match o with
 | Op.mk name args bbs regions attrs ty => 
    Op.mk name args bbs regions (attrs.add entry) ty

def BasicBlock.empty (name: String): BasicBlock := BasicBlock.mk name [] []
def BasicBlock.appendStmt (bb: BasicBlock) (stmt: BasicBlockStmt): BasicBlock := 
  match bb with
  | BasicBlock.mk name args bbs => BasicBlock.mk name args (bbs ++ [stmt])

def BasicBlock.appendStmts (bb: BasicBlock) (stmts: List BasicBlockStmt): BasicBlock := 
  match bb with
  | BasicBlock.mk name args bbs => BasicBlock.mk name args (bbs ++ stmts)

def Region.empty: Region := Region.mk [] 

def Region.appendBasicBlock (r: Region) (bb: BasicBlock) : Region := 
  coe (coe r ++ [bb])

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




-- | TODO: defunctionalize the lens?
@[reducible, simp]
abbrev lens' s a := ∀ {f: Type -> Type}, 
   [Functor f] -> [Inhabited (f s)] -> (a -> f a) -> (s -> f s)

def Op.lensSingletonRegion: lens' Op Region := 
 fun lens o =>
   match o with
   | Op.mk name args bbs [r] attrs ty => 
       Functor.map (fun r' => Op.mk name args bbs [r'] attrs ty) (lens r)
   | _ => panic! "expected op with single region: " ++ (doc o)


def Op.mutateSingletonRegion (o: Op) (f: Region -> Region): Op :=
 match o with
 | Op.mk name args bbs [r] attrs ty => Op.mk name args bbs [f r] attrs ty
 | _ => panic! "expected op with single region: " ++ (doc o)

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
