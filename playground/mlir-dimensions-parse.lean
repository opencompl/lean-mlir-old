import Lean
import Lean.Parser

open Lean
open Lean.Parser

def dimensionList : Parser :=
  let dimension := numLit  -- TODO: extend
  let p := sepByNoAntiquot (withAntiquotSpliceAndSuffix `sepBy dimension (symbol "x*")) (rawCh (trailingWs := true) 'x')
  p 

-- test
syntax "[dimensionList| " dimensionList "]" : term
macro_rules
  | `([dimensionList| $[$dims]x*]) => return quote dims

#check [dimensionList| 2x1]
#check [dimensionList| 2 x 1]

def dimensionListX : Parser :=
  let dimension := numLit  -- TODO: extend
  let p := sepByNoAntiquot (withAntiquotSpliceAndSuffix `sepBy dimension (symbol "foo")) (rawCh (trailingWs := true) 'x')
  p 


syntax "[dimensionListX| " dimensionListX "]" : term
macro_rules
  | `([dimensionListX| $[$dims]foo]) => return quote dims

#check [dimensionListX| 2x1]
#check [dimensionListX| 2 x 1]


def dimensionListY : Parser :=
  let dimension := numLit  -- TODO: extend
  let p := sepByNoAntiquot  (withAntiquotSpliceAndSuffix `sepBy dimension (symbol "foo")) (rawCh (trailingWs := true) 'y')
  p 


syntax "[dimensionListY| " dimensionListY "]" : term
macro_rules
  | `([dimensionListY| $[$dims]foo]) => return quote dims

#check [dimensionListY| 2y1]
#check [dimensionListY| 2 y 1]


def dimensionListZ : Parser :=
  let dimension := numLit  -- TODO: extend
  let p := sepByNoAntiquot (allowTrailingSep := true) (withAntiquotSpliceAndSuffix `sepBy dimension (symbol "foo")) (rawCh (trailingWs := true) 'y')
  p 


syntax "[dimensionListZ| " dimensionListZ "y" "]" : term
macro_rules
  | `([dimensionListZ| $[$dims]foo y]) => return quote dims

#check [dimensionListZ| 2y1y]
#check [dimensionListZ| 2 y 1]
