-- | PDL Dialect
-- | https://mlir.llvm.org/docs/Dialects/PDLOps/
import MLIR.EDSL
import MLIR.AST

open MLIR.AST

-- | https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlattribute-mlirpdlattributeop
syntax "pdl.attribute" (":" mlir_type)? (mlir_op_operand)? : mlir_op
syntax "pdl.erase" mlir_op_operand : mlir_op
syntax "pdl.operand" (":" mlir_type)? : mlir_op
syntax "pdl.operands" (":" mlir_type)? : mlir_op
syntax "pdl.operation" (str)? "(" sepBy(mlir_op_operand, ",") ")" ("->" "(" sepBy(mlir_op_operand, ",") ":" sepBy(mlir_type, ","))? : mlir_op
-- | TODO: don't use term, use something more precise like `intLit`.
syntax "pdl.pattern" (str)? ":" "benifit" "(" term ")"  mlir_region : mlir_op
syntax "pdl.replace" mlir_op_operand "with"
       ("(" sepBy(mlir_op_operand, ",")  ":" sepBy(mlir_type, ",")  ")")?
  ("(" mlir_op_operand ")")? : mlir_op
-- | TODO: replace `term` with something much more precise.
-- https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresult-mlirpdlresultop
syntax "pdl.result" term "of" mlir_op_operand : mlir_op
-- https://mlir.llvm.org/docs/Dialects/PDLOps/#pdlresults-mlirpdlresultsop
syntax "pdl.results" (term)? "of" mlir_op ("->" mlir_type)? : mlir_op
--
syntax "pdl.type" (":" mlir_type) : mlir_op
syntax "pdl.types" (":" "[" sepBy(mlir_type, ",") "]")? : mlir_op


set_option hygiene false in -- need to disable hygiene for i32 expansion.
macro_rules
  | `([mlir_op| pdl.replace $op with ($args,* : $tys,*)]) =>
        `( [mlir_op| "pdl.replace" ($op, $args,* ) : () -> ()])

