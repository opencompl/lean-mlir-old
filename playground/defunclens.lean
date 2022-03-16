#check ULift


/-
inductive NoWorkBlockLens: Type -> Type where
| BlockOp: o -> NoWorkBlockLens o
| BlockUnit:  NoWorkBlockLens Unit
-/

-- | Needs to be sure that o and Unit are on the same level.
inductive BlockLens: Type u -> Type (max 0 (u + 1))  where
| BlockOp: (o: Type u) -> BlockLens o
| BlockUnit:  BlockLens (PUnit)
