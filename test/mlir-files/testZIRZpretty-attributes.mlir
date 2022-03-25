"builtin.module"() ({
  "test.dense_attr"() {foo.dense_attr = dense<[1, 2, 3]> : tensor<3 × i32>} : () -> ()
  "test.non_elided_dense_attr"() {foo.dense_attr = dense<[1, 2]> : tensor<2 × i32>} : () -> ()
  "test.sparse_attr"() {foo.sparse_attr = sparse<[[0, 0, 5]], -2.000000e+00> : vector<1 × 1 × 10 × f16>} : () -> ()
  "test.opaque_attr"() {foo.opaque_attr = opaque<"elided_large_const", "0 × EBFE"> : tensor<100 × f32>} : () -> ()
  "test.dense_splat"() {foo.dense_attr = dense<1> : tensor<3 × i32>} : () -> ()
}) : () -> ()

// -----
