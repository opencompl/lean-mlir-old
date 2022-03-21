import MLIR.StdDialect 
import MLIR.EDSL 
import MLIR.StdDialect

open MLIR.Doc
open MLIR.AST
open MLIR.EDSL


-- End to end example of using the einstein summation convention frontend
-- to generate linalg code.

-- | define expressions via EDSL

-- | Mix into MLIR definitions.
def complex0 := [mlir_op|
   func @"main"()  {    
     %c42 = "std.constant" () { value = 42 : i32} : i32
     %x = "complex.create" (%c42, %c42) : (i32, i32) -> !"complex" 
     %y = "complex.mul" (%x, %x) : (!"complex", !"complex") -> !"complex" 
     %z = "complex.exp" (%y) : (!"complex") -> !"complex"
    "std.return"(%z) : !"complex" -> ()
  }
]


-- | TODO: this should be in StdDialect or something
syntax "constant" num ":" mlir_type : mlir_op
syntax "return" mlir_op_operand ":" mlir_type : mlir_op


macro_rules
| `([mlir_op| constant $x:numLit : $t:mlir_type]) => 
        `([mlir_op| "TODO_constant" () : () ])

macro_rules
| `([mlir_op| return $x:mlir_op_operand : $t:mlir_type]) => 
        `([mlir_op| "TODO_return" () : () ])

-- Complex numbers MLIR encoding
syntax mlir_op_operand "+" mlir_op_operand "i" : mlir_op
syntax mlir_op_operand "c*" mlir_op_operand :  mlir_op
syntax mlir_op_operand "c*" mlir_op_operand :  mlir_op
syntax "e^" mlir_op_operand :  mlir_op

macro_rules
| `([mlir_op| $x:mlir_op_operand + $y:mlir_op_operand i]) => 
        `([mlir_op| "complex.create" ($x, $y) : () -> !"complex"  ])

macro_rules
| `([mlir_op| $x:mlir_op_operand c* $y:mlir_op_operand]) => 
        `([mlir_op| "complex.mul" ($x, $y) : () -> !"complex"  ])

macro_rules
| `([mlir_op| e^ $x:mlir_op_operand]) => 
        `([mlir_op| "complex.exp" ($x) : () -> !"complex"  ])


-- | complex example
def complex1 := [mlir_op|
   func @"main"()  {    
     %c42 = constant 42 : i32
     %x = %c42 + %c42 i
     %y = %x c* %x 
     %z = e^ %y
    return %z : !"complex"
  }]



-- Complex numbers DSL encoding
-- | TODO: make this generate MLIR.

declare_syntax_cat complex
syntax complex "*" complex : complex
syntax num "+" num "i"  : complex
syntax "e^"  complex : complex
syntax "(" complex ")" : complex
syntax "[complex|" complex "]" : term
syntax term : mlir_op_operand


structure GenM (a: Type) where 
  gen: (Int × List BasicBlockStmt) → (a × Int  ×List BasicBlockStmt)

instance : Functor GenM where 
  map f gen := 
  { gen := fun input => 
    let (x, stmts) := gen.gen input
    (f x, stmts)
  }

instance : Monad GenM where
  pure v := { gen := fun (x, stmts) => (v, x, stmts) } 
  bind gena a2genb  := {
    gen := fun (count, ops) => 
        let (a, count, ops) := gena.gen (count, ops)
        let genb := a2genb a
        let (b, count, ops) := genb.gen (count, ops)
        (b, count, ops)
  }
  -- |applicative 
  seq gena2b unit2gena := {
    gen := fun (count, ops) =>
      let (a2b, count, ops) := (gena2b).gen (count, ops)
      let (a, count, ops) := (unit2gena ()).gen (count, ops)
      (a2b a, count, ops)
  }

def runGenM (g: GenM α) (name: String := "entry") (args: List (SSAVal ×MLIRTy) := []): BasicBlock := 
  BasicBlock.mk name args (g.gen (0, [])).snd.snd

-- | append an op with a named lh
def appendOp (op: Op): GenM SSAVal :=
  { gen := fun (x, stmts) => 
       let name := SSAVal.SSAVal ("v" ++ toString x)
       let stmt := BasicBlockStmt.StmtAssign name op
       (name, x + 1,  stmts ++ [stmt])
  }

-- | append an op with no name
def appendOp_ (op: Op): GenM Unit := 
  { gen := fun (x, stmts) => 
     let stmt := BasicBlockStmt.StmtOp op
     ((), x,  stmts ++ [stmt])
}


macro_rules
| `([complex| $x:numLit + $y:numLit i]) => 
    `(do
        let k <- appendOp [mlir_op| constant $x : i32]
        let l <- appendOp [mlir_op| constant $y : i32]
        appendOp [mlir_op| [escape| k] + [escape| l] i]
     )

macro_rules
| `([complex| ($x:complex)]) => `([complex| $x])



set_option maxRecDepth 100
def complex2 := runGenM [complex| 42 + 42 i]
#eval IO.eprintln $ Pretty.doc $ complex2

macro_rules
| `([complex| $x:complex * $y:complex]) => 
    `(do
        let k <- [complex| $x]
        let l <- [complex| $y]
        appendOp [mlir_op| [escape| k] c* [escape| l]]
     )

def complex3 := runGenM [complex| (1 + 2 i) * (3 + 4 i)]
#eval IO.eprintln $ Pretty.doc $ complex3


-- def complex2 := [complex| e^((42 + 42 i) * (42 + 42 i))]