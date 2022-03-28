#loc = loc(callsite("foo" at "mysource.cc":10:8))
#map0 = affine_map<() -> (0)>
#map1 = affine_map<() -> (8)>
#set = affine_set<(d0) : (1 == 0)>
"builtin.module"() ({
  "func.func"() ({
    %0 = "foo"() : () -> i32
    %1 = "arith.constant"() {value = 4 : index} : () -> index
    "affine.for"() ({
    ^bb0(%arg0: index):
      "affine.yield"() : () -> ()
    }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
    "affine.if"(%1) ({
      "affine.yield"() : () -> ()
    }, {
    }) {condition = #set} : (index) -> ()
    "affine.if"(%1) ({
      "affine.yield"() : () -> ()
    }, {
    }) {condition = #set} : (index) -> ()
    "func.return"(%0) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "inline_notation"} : () -> ()
  "func.func"() ({
  }) {arg_attrs = [{foo.loc_attr = #loc}], function_type = (i1) -> (), sym_name = "loc_attr", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "foo"() : () -> ()
    "foo"() : () -> ()
    "foo"() : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "escape_strings"} : () -> ()
  "foo.op"() : () -> ()
  "func.func"() ({
  ^bb0(%arg0: i32, %arg1: i64):
    "func.return"() : () -> ()
  }) {function_type = (i32, i64) -> (), sym_name = "argLocs"} : () -> ()
  "foo.unknown_op_with_bbargs"() ({
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    %0 = "arith.addi"(%arg0, %arg1) : (i32, i32) -> i32
    "foo.yield"(%0) : (i32) -> ()
  }) : () -> ()
  "func.func"() ({
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "location_name_child_is_name"} : () -> ()
}) : () -> ()


