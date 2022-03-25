"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym.new_name = "replaced_foo", sym_name = "symbol_foo", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "foo.op"() {non_symbol_attr, use = [{nested_symbol = [@symbol_foo], other_use = @symbol_bar, z_use = @symbol_foo}], z_non_symbol_attr_3} : () -> ()
  }) {function_type = () -> (), sym.use = @symbol_foo, sym_name = "symbol_bar"} : () -> ()
  "builtin.module"() ({
    "foo.op"() {test.nested_reference = @symbol_foo} : () -> ()
  }) {test.reference = @symbol_foo} : () -> ()
}) {sym.outside_use = @symbol_foo} : () -> ()

// -----
"builtin.module"() ({
  "builtin.module"() ({
    "func.func"() ({
    }) {function_type = () -> (), sym.new_name = "replaced_foo", sym_name = "foo", sym_visibility = "nested"} : () -> ()
  }) {sym_name = "module_a"} : () -> ()
  "builtin.module"() ({
    "builtin.module"() ({
      "func.func"() ({
      }) {function_type = () -> (), sym.new_name = "replaced_foo", sym_name = "foo", sym_visibility = "nested"} : () -> ()
    }) {sym.new_name = "replaced_module_c", sym_name = "module_c"} : () -> ()
  }) {sym.new_name = "replaced_module_b", sym_name = "module_b"} : () -> ()
  "func.func"() ({
    "foo.op"() {use_1 = @module_a::@foo, use_2 = @module_b::@module_c::@foo} : () -> ()
  }) {function_type = () -> (), sym_name = "symbol_bar"} : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym.new_name = "replaced_name", sym_name = "failed_repl", sym_visibility = "private"} : () -> ()
  "foo.possibly_unknown_symbol_table"() ({
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = () -> (), sym.new_name = "replaced_foo", sym_name = "symbol_foo", sym_visibility = "private"} : () -> ()
  "func.func"() ({
    "foo.op"() {non_symbol_attr, use = [#test.sub_elements_access<[@symbol_foo], @symbol_bar, @symbol_foo>], z_non_symbol_attr_3} : () -> ()
  }) {function_type = () -> (), sym_name = "symbol_bar"} : () -> ()
}) : () -> ()

// -----
