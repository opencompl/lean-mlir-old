-- Tasty frontend with dependent typing

import MLIR.AST
import MLIR.Doc
open MLIR.Doc
open MLIR.AST

def FinList (n: Nat) (α: Type) := Fin n -> α 

def finList0 {α : Type}: FinList 0 α := fun n => Fin.elim0 n 

instance : Coe (α × α) (FinList 2 α) where
   coe as := 
   match as with
   | (a0, a1) =>
       fun n => match n with
                | ⟨0, _ ⟩ => a0
                | ⟨1, _ ⟩ => a1


instance : Coe (α) (FinList 1 α) where
   coe a := fun n => match n with
                | ⟨0, _ ⟩ => a

def foo : FinList 2 Nat := (2, 3)


def finBump {n: Nat} (f: Fin n): Fin (Nat.succ n) := 
  { val := f.val, isLt := Nat.lt_succ_of_le (Nat.le_of_lt f.isLt) }


def fins (n: Nat) : List (Fin n) :=
 match n with
 | 0 => []
 | Nat.succ n' => [Fin.ofNat n'] ++ ((fins n').map finBump)

instance : Coe (FinList n α) (List α) where
  coe xs := (fins n).map xs

-- inductive OpT (OpKind: Type) (Ops: OpKind -> Nat) (BBs: OpKind -> Nat)                                                     │~
--      (Regions: OpKind -> Nat): Type:=                                                                                        │~
--   | mk: (k: OpKind)                                                                                                          │~
--        -> (args: FinList (Ops k) SSAVal)                                                                                     │~
--        -> (bbs: FinList (BBs k) BBName)                                                                                      │~
--        -> (regions: FinList (Regions k) Region)                                                                              │~
--        -> (attr: AttrDict)                                                                                                   │~
--        -> (ty: MLIRTy)                                                                                                       │~
--        -> OpT OpKind Ops BBs Regionsnductive OpT (OpKind: Type) (Ops: OpKind -> Nat) (BBs: OpKind -> Nat)
--    (Regions: OpKind -> Nat): Type:= 
-- | mk: (k: OpKind) 
--      -> (args: FinList (Ops k) SSAVal) 
--      -> (bbs: FinList (BBs k) BBName)
--      -> (regions: FinList (Regions k) Region)
--      -> (attr: AttrDict)
--      -> (ty: MLIRTy)
--      -> OpT OpKind Ops BBs Regions



class  ToList (f: Type -> Type) where
   toList : f α → List α

instance  [ToList f] : Coe (f α) (List α) where
  coe := ToList.toList


-- def OpT.op {OpKind: Type}
--     {Ops: OpKind -> Nat}
--     {BBs: OpKind -> Nat}
--     {Regions: OpKind -> Nat}
--     [ToString OpKind]
--     (op: OpT OpKind Ops BBs Regions): Op :=
--     match op with
--     | OpT.mk k args bbs rgns attr ty => Op.mk (toString k) args bbs rgns attr ty
  

-- instance {OpKind: Type}
--     {Ops: OpKind -> Nat}
--     {BBs: OpKind -> Nat}
--     {Regions: OpKind -> Nat}
--     {kind: OpKind}
--     [Coe OpKind String]
--     : Coe (OpT OpKind Ops BBs Regions) Op where
--   coe op := 
--     match op with
--     | OpT.mk k args bbs rgns attr ty => Op.mk k args bbs rgns attr ty



-- namespace Example
-- inductive LLVMOps  -- | declare ops
-- | Add | Not | Constant | Branch 

-- instance : ToString LLVMOps where
--   toString k := 
--    match k with
--    | LLVMOps.Add => "add" | LLVMOps.Not => "not" | LLVMOps.Constant => "const" | LLVMOps.Branch => "br"
   
-- @[simp]
-- def llvmOpArgs (kind: LLVMOps): Nat := -- | declare number of args
-- match kind with
-- | LLVMOps.Add => 2 | LLVMOps.Not => 1 | LLVMOps.Constant => 0 | LLVMOps.Branch => 0

-- @[simp]
-- def llvmOpBBs (kind: LLVMOps) : Nat := -- | declare number of bb args
-- match kind with
-- | LLVMOps.Branch => 1 | _ => 0

-- def llvmOpRegions (kind: LLVMOps): Nat := 0 -- | declare number of regions


-- -- | create a typed llvm operation, explicit number of args, bb args region args.
-- abbrev llvmOpT: Type := OpT LLVMOps llvmOpArgs llvmOpBBs llvmOpRegions
-- #reduce llvmOpT

-- def llvm_add: Op := -- | untyped llvm add, unchecked number of args
--   Op.mk "add"  [SSAVal.SSAVal "x", SSAVal.SSAVal "y"] [] [] (AttrDict.mk []) (MLIRTy.int 32)
-- #eval IO.eprintln $ Pretty.doc $ llvm_add

-- def llvm_addT : llvmOpT :=  -- | typed LLVM op, checked number of args (2 args)
--   let args : FinList (llvmOpArgs LLVMOps.Add) SSAVal := by 
--     simp;
--     apply (coe (SSAVal.SSAVal "x", SSAVal.SSAVal "y"));
--   OpT.mk LLVMOps.Add  args finList0 finList0 (AttrDict.mk []) (MLIRTy.int 32)
-- #eval IO.eprintln $ Pretty.doc $ llvm_addT.op

-- def llvm_add_one_arg: Op := -- | ERROR: incorrect number of args
--   Op.mk "add"  [SSAVal.SSAVal "x"] [] [] (AttrDict.mk []) (MLIRTy.int 32)
-- #eval IO.eprintln $ Pretty.doc $ llvm_add_one_arg

-- /-
-- def llvm_add_one_argT : llvmOpT := -- | ERROR not possible, thanks to dependent types. 
--   let args : FinList (llvmOpArgs LLVMOps.Add) SSAVal := by 
--     simp;
--     apply (coe (SSAVal.SSAVal "x"));
--   OpT.mk LLVMOps.Add  args finList0 finList0 (AttrDict.mk []) (MLIRTy.int 32)
-- -/
-- end Example
