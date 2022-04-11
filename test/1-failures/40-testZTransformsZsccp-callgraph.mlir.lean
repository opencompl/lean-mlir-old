
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
  ^bb0(%arg0: i32):
    "func.return"(%arg0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "private", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "func.call"(%0) {callee = @private} : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "simple_private"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    "func.return"(%arg0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "nested", sym_visibility = "nested"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "func.call"(%0) {callee = @nested} : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "simple_nested"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "builtin.module"() ({
    "func.func"() ({
    ^bb0(%arg0: i32):
      %0 = "arith.constant"() {value = 1 : i32} : () -> i32
      "func.return"(%0, %arg0) : (i32, i32) -> ()
    }) {function_type = (i32) -> (i32, i32), sym_name = "nested", sym_visibility = "nested"} : () -> ()
    "func.func"() ({
      %0 = "arith.constant"() {value = 1 : i32} : () -> i32
      %1:2 = "func.call"(%0) {callee = @nested} : (i32) -> (i32, i32)
      "func.return"(%1#0, %1#1) : (i32, i32) -> ()
    }) {function_type = () -> (i32, i32), sym_name = "nested_not_all_uses_visible"} : () -> ()
  }) {sym_name = "nested_module", sym_visibility = "public"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    "func.return"(%0, %arg0) : (i32, i32) -> ()
  }) {function_type = (i32) -> (i32, i32), sym_name = "public"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1:2 = "func.call"(%0) {callee = @public} : (i32) -> (i32, i32)
    "func.return"(%1#0, %1#1) : (i32, i32) -> ()
  }) {function_type = () -> (i32, i32), sym_name = "simple_public"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    "func.return"(%0, %arg0) : (i32, i32) -> ()
  }) {function_type = (i32) -> (i32, i32), sym_name = "callable", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1:2 = "func.call"(%0) {callee = @callable} : (i32) -> (i32, i32)
    "func.return"(%1#0, %1#1) : (i32, i32) -> ()
  }) {function_type = () -> (i32, i32), sym_name = "non_call_users"} : () -> ()
  "live.user"() {uses = [@callable]} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    "unknown.return"(%arg0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "callable", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "func.call"(%0) {callee = @callable} : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "unknown_terminator"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    "func.return"(%arg0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "callable", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "arith.constant"() {value = 2 : i32} : () -> i32
    %2 = "func.call"(%0) {callee = @callable} : (i32) -> i32
    %3 = "func.call"(%1) {callee = @callable} : (i32) -> i32
    "func.return"(%2, %3) : (i32, i32) -> ()
  }) {function_type = () -> (i32, i32), sym_name = "conflicting_constant"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    "unknown.return"(%arg0) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "callable", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "func.call"(%0) {callee = @callable} : (i32) -> i32
    %2 = "func.call"(%arg0) {callee = @callable} : (i32) -> i32
    "func.return"(%1, %2) : (i32, i32) -> ()
  }) {function_type = (i32) -> (i32, i32), sym_name = "conflicting_constant"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 20 : i32} : () -> i32
    %1 = "arith.cmpi"(%arg0, %0) {predicate = 6 : i64} : (i32, i32) -> i1
    "cf.cond_br"(%1)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  
    %2 = "arith.constant"() {value = 1 : i32} : () -> i32
    "func.return"(%2) : (i32) -> ()
  ^bb2:  
    %3 = "arith.constant"() {value = 1 : i32} : () -> i32
    %4 = "arith.addi"(%arg0, %3) : (i32, i32) -> i32
    "func.return"(%4) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "complex_inner_if", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> i1, sym_name = "complex_cond", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "func.call"() {callee = @complex_cond} : () -> i1
    "cf.cond_br"(%0)[^bb1, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3 × i32>} : (i1) -> ()
  ^bb1:  
    "func.return"(%arg0) : (i32) -> ()
  ^bb2:  
    %1 = "func.call"(%arg0) {callee = @complex_inner_if} : (i32) -> i32
    %2 = "func.call"(%1) {callee = @complex_callee} : (i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "complex_callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() {value = 1 : i32} : () -> i32
    %1 = "func.call"(%0) {callee = @complex_callee} : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = (i32) -> i32, sym_name = "complex_caller"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "test.functional_region_op"() ({
      %2 = "arith.constant"() {value = 1 : i32} : () -> i32
      "test.return"(%2) : (i32) -> ()
    }) : () -> (() -> i32)
    %1 = "func.call_indirect"(%0) : (() -> i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "non_symbol_defining_callable"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = true} : () -> i1
    %1 = "arith.constant"() {value = 0 : i32} : () -> i32
    %2 = "arith.constant"() {value = 1 : i32} : () -> i32
    %3 = "arith.select"(%0, %1, %2) : (i1, i32, i32) -> i32
    "func.return"(%3) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "unreferenced_private_function", sym_visibility = "private"} : () -> ()
}) : () -> ()



] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
    FS.writeFile "testZTransformsZsccp-callgraph.out.txt" str
