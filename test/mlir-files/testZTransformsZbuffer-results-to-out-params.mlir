"builtin.module"() ({
  "func.func"() ({
    %0 = "test.source"() : () -> memref<f32>
    "func.return"(%0) : (memref<f32>) -> ()
  }) {function_type = () -> memref<f32>, sym_name = "basic"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: memref<1 × f32>):
    %0 = "test.source"() : () -> memref<2 × f32>
    "func.return"(%0) : (memref<2 × f32>) -> ()
  }) {function_type = (memref<1 × f32>) -> memref<2 × f32>, sym_name = "presence_of_existing_arguments"} : () -> ()
  "func.func"() ({
    %0:2 = "test.source"() : () -> (memref<1 × f32>, memref<2 × f32>)
    "func.return"(%0#0, %0#1) : (memref<1 × f32>, memref<2 × f32>) -> ()
  }) {function_type = () -> (memref<1 × f32>, memref<2 × f32>), sym_name = "multiple_results"} : () -> ()
  "func.func"() ({
    %0:3 = "test.source"() : () -> (i1, memref<f32>, i32)
    "func.return"(%0#0, %0#1, %0#2) : (i1, memref<f32>, i32) -> ()
  }) {function_type = () -> (i1, memref<f32>, i32), sym_name = "non_memref_types"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> memref<f32>, sym_name = "external_function", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> memref<f32>, res_attrs = [{test.some_attr}], sym_name = "result_attrs", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (memref<1 × f32>, memref<2 × f32>, memref<3 × f32>), res_attrs = [{}, {test.some_attr}, {}], sym_name = "mixed_result_attrs", sym_visibility = "private"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> memref<1 × f32>, sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "func.call"() {callee = @callee} : () -> memref<1 × f32>
    "test.sink"(%0) : (memref<1 × f32>) -> ()
    "func.return"() : () -> ()
  }) {function_type = () -> (), sym_name = "call_basic"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (memref<1 × f32>, memref<2 × f32>), sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0:2 = "func.call"() {callee = @callee} : () -> (memref<1 × f32>, memref<2 × f32>)
    "test.sink"(%0#0, %0#1) : (memref<1 × f32>, memref<2 × f32>) -> ()
  }) {function_type = () -> (), sym_name = "call_multiple_result"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (i1, memref<1 × f32>, i32), sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0:3 = "func.call"() {callee = @callee} : () -> (i1, memref<1 × f32>, i32)
    "test.sink"(%0#0, %0#1, %0#2) : (i1, memref<1 × f32>, i32) -> ()
  }) {function_type = () -> (), sym_name = "call_non_memref_result"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> memref<? × f32>, sym_name = "callee", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    %0 = "func.call"() {callee = @callee} : () -> memref<? × f32>
    "test.sink"(%0) : (memref<? × f32>) -> ()
  }) {function_type = () -> (), sym_name = "call_non_memref_result"} : () -> ()
}) : () -> ()

// -----
