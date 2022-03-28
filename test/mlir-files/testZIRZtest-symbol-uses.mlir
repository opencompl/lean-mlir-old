"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "symbol_foo", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "foo.op"() {non_symbol_attr, use = [{nested_symbol = [@symbol_foo]}], z_other_non_symbol_attr} : () -> ()
  }) {function_type = () -> (), sym.use = @symbol_foo, sym_name = "symbol_bar"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "symbol_removable", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = () -> (), sym_name = "symbol_baz", sym_visibility = "private"} : () -> ()
  "builtin.module"() ({
    "foo.op"() {test.nested_reference = @symbol_baz} : () -> ()
  }) {test.reference = @symbol_baz} : () -> ()
}) {sym.outside_use = @symbol_foo} : () -> ()


"builtin.module"() ({
  "builtin.module"() ({
    "builtin.module"() ({
      "func.func"() ({
      }) {function_type = () -> (), sym_name = "foo", sym_visibility = "nested"} : () -> ()
    }) {sym_name = "module_c"} : () -> ()
  }) {sym_name = "module_b"} : () -> ()
  "func.func"() ({
    "foo.op"() {use_1 = [{nested_symbol = [@module_b::@module_c::@foo]}], use_2 = @module_b} : () -> ()
  }) {function_type = () -> (), sym_name = "symbol_bar"} : () -> ()
}) : () -> ()


"builtin.module"() ({
  "func.func"() ({
    "foo.possibly_unknown_symbol_table"() ({
    }) : () -> ()
  }) {function_type = () -> (), sym_name = "symbol_bar"} : () -> ()
}) : () -> ()


