"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 10 : index} : () -> index
    "scf.for"(%1, %2, %1) ({
    ^bb0(%arg0: index):
      %3 = "use0"(%arg0) : (index) -> i1
      "scf.if"(%3) ({
        "use1"(%arg0) : (index) -> ()
        "scf.yield"() : () -> ()
      }, {
        "use2"(%arg0) : (index) -> ()
        "scf.yield"() : () -> ()
      }) : (i1) -> ()
      "use3"(%arg0) : (index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "structured_cfg"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 0 : index} : () -> index
    %1 = "arith.constant"() {value = 1 : index} : () -> index
    %2 = "arith.constant"() {value = 10 : index} : () -> index
    "scf.for"(%1, %2, %1) ({
    ^bb0(%arg0: index):
      "test.two_region_op"() ({
        "work"() : () -> ()
      }, {
        "work"() : () -> ()
      }) : () -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "correct_number_of_regions"} : () -> ()
}) : () -> ()


