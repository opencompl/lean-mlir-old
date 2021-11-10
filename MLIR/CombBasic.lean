-- Ideas from: https://www.cs.utah.edu/~regehr/papers/pldi15.pdf

import Lean.Parser
import Lean.Parser.Extra
open Lean
open Lean.Parser

abbrev Reg := Int

inductive Cond
  | eq
  | ne

inductive Op : Type where
  | const: Int -> Op -- constant <value>
  | add: Reg -> Reg -> Op -- add <lhs> <rhs>
  | mul: Reg -> Reg -> Op -- mul <lhs> <rhs>
  | select: Reg -> Reg -> Reg -> Op -- select <c> <t> <f>
  | icmp: Cond -> Reg -> Reg -> Op -- icmp <cmp> <lhs> <rhs>

open Op

-- Stmt (dest, <operands>)
inductive Stmt : Type where
 | setop: Reg -> Op -> Stmt -- %reg = op
 | unreachable: Stmt

open Stmt

abbrev Program := List Stmt
abbrev Env := Reg -> Option Int

def evalOp (op : Op) (env: Env) : Option Int :=
  match op with
  | const x => some x
  | add lhs rhs =>
    match env lhs with
      | none => none
      | some lv =>
        match env rhs with
        | none => none
        | some rv => some (lv + rv)
  | _ => none

def evalStmt (s : Stmt) (e : Env) : Option Env :=
  match s with
  | setop res op =>
    match evalOp op e with
    | some val => some (λ r => if r = res then val else e r)
    | none => none
  | unreachable => none

def eval (p : Program) (e : Env) : Option Env := do
  match p with
  | [] => some e
  | s :: p' =>
    match evalStmt s e with
    | some e' => eval p' e'
    | none => none

def program_equiv (p: Program) (q: Program) : Prop := sorry