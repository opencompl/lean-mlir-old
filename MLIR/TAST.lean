-- Tasty frontend with dependent typing

import MLIR.AST
open MLIR.AST

def FinList (n: Nat) (α: Type) := Fin n -> α 

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

inductive OpT (OpKind: Type) (Ops: OpKind -> Type -> Type) (BBs: OpKind -> Type -> Type)
   (Regions: OpKind -> Type -> Type): Type:= 
| mk: (k: OpKind) 
     -> (args: Ops k SSAVal) 
     -> (bbs: BBs k BBName)
     -> (regions: Regions k Region)
     -> (attr: AttrDict)
     -> (ty: MLIRTy)
     -> OpT OpKind Ops BBs Regions

inductive RegionT (OpKind: Type) (Ops: OpKind -> Type -> Type) (BBs: OpKind -> Type -> Type) (Regions: OpKind -> Type -> Type) : Type := 
| RegionNil: RegionT

class  ToList (f: Type -> Type) where
   toList : f α → List α

instance  [ToList f] : Coe (f α) (List α) where
  coe := ToList.toList
  

instance {OpKind: Type}
    {Ops: OpKind -> Type -> Type}
    {BBs: OpKind -> Type -> Type}
    {Regions: OpKind -> Type -> Type}
    {kind: OpKind}
    [Coe OpKind String]
    [∀ (k: OpKind), ToList (Ops k)]
    [∀ (k: OpKind), ToList (BBs k)]
    [∀ (k: OpKind), ToList (Regions k)]: Coe (OpT OpKind Ops BBs Regions) Op where
  coe op := 
    match op with
    | OpT.mk k args bbs rgns attr ty => Op.mk k args bbs rgns attr ty


