"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    %0:2 = "new_processor_id_and_range"() : () -> (index, index)
    "scf.for"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: index):
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index) -> (), sym_name = "map1d"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    %0:2 = "new_processor_id_and_range"() : () -> (index, index)
    %1:2 = "new_processor_id_and_range"() : () -> (index, index)
    "scf.for"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: index):
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (index, index, index) -> (), sym_name = "map2d"} : () -> ()
}) : () -> ()


