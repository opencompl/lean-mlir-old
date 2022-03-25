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

// -----
"builtin.module"() ({
  "func.func"() ({
    "regionOp0"() ({
      "op0"() : () -> ()
      "cf.br"()[^bb2] : () -> ()
    ^bb1:  // no predecessors
      "op1"() : () -> ()
      "cf.br"()[^bb2] : () -> ()
    ^bb2:  // 2 preds: ^bb0, ^bb1
      "op2"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "unstructured_cfg"} : () -> ()
}) : () -> ()

// -----
