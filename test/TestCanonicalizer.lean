
import MLIR.Doc
import MLIR.AST
import MLIR.EDSL

open  MLIR.EDSL
open MLIR.Doc
open IO

-- | write an op into the path
def o: Op := [mlir_op|
"module"() ( {
  "func"() ( {
  ^bb0(%arg0: i32):  
    -- %0 = "memref.alloc"() {operand_segment_sizes = dense<0> : vector<2×i32>, test.ptr = "alloc"} : () -> memref<8× 64×f32>
    -- "memref.dealloc"(%0) {test.ptr = "dealloc"} : (memref<8×64×f32>) -> ()
    -- "std.return"() {test.ptr = "return"} : () -> ()
  }) {sym_name = "no_side_effects", test.ptr = "func", type = (memref<2×f32>) -> ()} : () -> ()
  "module_terminator"() : () -> ()
}) : () -> ()


] 
-- | main program
def main : IO Unit :=
    let str := Pretty.doc o
    FS.writeFile "test-alias-analysis-modref.out.txt" str
