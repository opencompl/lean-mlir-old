
import MLIR.Doc
import MLIR.AST
import MLIR.EDSL

open Lean
open Lean.Parser
open  MLIR.EDSL
open MLIR.AST
open MLIR.Doc
open IO

set_option maxHeartbeats 999999999


declare_syntax_cat mlir_ops
syntax (ws mlir_op ws)* : mlir_ops
syntax "[mlir_ops|" mlir_ops "]" : term

macro_rules
|`([mlir_ops| $[ $xs ]* ]) => do 
  let xs <- xs.mapM (fun x =>`([mlir_op| $x]))
  quoteMList xs.toList (<-`(MLIR.AST.Op))

  
-- | write an op into the path
def o: List Op := [mlir_ops|
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() {interrupt_before_all = true} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() ({
      "bar"() : () -> ()
    }) {interrupt_after_all = true} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() ({
      "bar0"() : () -> ()
    }, {
      "bar1"() : () -> ()
    }) {interrupt_after_region = 0 : i64} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "foo"() : () -> ()
    "test.two_region_op"() ({
      "work"() : () -> ()
    }, {
      "work"() : () -> ()
    }) {interrupt_after_all = true} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "foo"() : () -> ()
    "test.two_region_op"() ({
      "work"() : () -> ()
    }, {
      "work"() : () -> ()
    }) {interrupt_after_region = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() ({
      "bar0"() : () -> ()
    }, {
      "bar1"() : () -> ()
    }) {skip_before_all = true} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() ({
      "bar0"() : () -> ()
    }, {
      "bar1"() : () -> ()
    }) {skip_after_all = true} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: f32):
    %0 = "foo"() ({
      "bar0"() : () -> ()
    }, {
      "bar1"() : () -> ()
    }) {skip_after_region = 0 : i64} : () -> f32
    %1 = "arith.addf"(%0, %arg0) : (f32, f32) -> f32
    "func.return"(%1) : (f32) -> ()
  }) {function_type = (f32) -> f32, sym_name = "main"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZIRZgeneric-visitors-interrupt.out.txt" str
