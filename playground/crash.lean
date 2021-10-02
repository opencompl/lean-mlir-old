declare_syntax_cat inum
syntax "i"num : inum

syntax "inum% " inum : term
macro_rules 
| `(inum% i$x:numLit) => `($x + 42)

-- def fooNoSpace : Int := (inum% i10)
def foo : Int := (inum% i 10)
#print foo



declare_syntax_cat ioptional
syntax "<[" num ? "]>": ioptional

macro_rules 
| `(ioptional% $x:numLit) => `(10)

-- def bar := optional% <[ ]>
-- #print bar

def bar2 := (ioptional% <[ 10 ]>)
#print bar2

