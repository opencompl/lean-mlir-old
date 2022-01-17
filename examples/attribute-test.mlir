// Check if it is legal to define attributes inside any region.
// It is not legal. So the toplevel module parser is in fact special!

func @main(%A:memref<?x?xf32>, %B: memref<?x?xf32>, %C:memref<?x?xf32>) {
  #matmul_accesses = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (k, n)>,
    affine_map<(m, n, k) -> (m, n)>
  ]

  #matmul_trait = {
    doc = "C(m, n) += A(m, k) * B(k, n)",
    indexing_maps = #matmul_accesses,
    library_call = "linalg_matmul",
    iterator_types = ["parallel", "parallel", "reduction"]
  }
  return
}


