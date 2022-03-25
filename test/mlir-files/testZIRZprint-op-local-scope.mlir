#map = affine_map<(d0) -> (d0 * 2)>
"builtin.module"() ({
  %0 = "foo.op"() : () -> memref<?xf32, #map>
}) : () -> ()

// -----
