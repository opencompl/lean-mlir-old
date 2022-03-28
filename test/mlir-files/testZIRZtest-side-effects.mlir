#map = affine_map<(d0, d1) -> (d1, d0)>
"builtin.module"() ({
  %0 = "test.side_effect_op"() : () -> i32
  %1 = "test.side_effect_op"() {effects = [{effect = "read"}, {effect = "free"}]} : () -> i32
  %2 = "test.side_effect_op"() {effects = [{effect = "write", test_resource}]} : () -> i32
  %3 = "test.side_effect_op"() {effects = [{effect = "allocate", on_result, test_resource}]} : () -> i32
  %4 = "test.side_effect_op"() {effects = [{effect = "read", on_reference = @foo_ref, test_resource}]} : () -> i32
  %5 = "test.side_effect_op"() {effect_parameter = #map} : () -> i32
  %6 = "test.unregistered_side_effect_op"() {effect_parameter = #map} : () -> i32
}) : () -> ()


