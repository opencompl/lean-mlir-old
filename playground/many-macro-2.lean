-- | key value pairs of numbers
declare_syntax_cat num_kv
syntax term "::" term : num_kv

syntax "num_kv% " num_kv : term
macro_rules
 | `( num_kv% $k :: $v ) => `( ($k , $v) )

def test_kv := (num_kv% 1 :: 2)
#print test_kv


-- -- | list of key pairs
-- declare_syntax_cat dict
-- syntax "<[" sepBy(num_kv, ",") "]>" : dict

-- syntax "dict%" dict : term

-- macro_rules
-- | `(dict% $kv,* ) => `()



-- def foo : List (Int × Int) := (dict% <[ 1 :: 2 , 3 :: 4 ]>)
-- #print foo
