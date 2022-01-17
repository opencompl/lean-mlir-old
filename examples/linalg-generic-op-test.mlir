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

func @main(%A:memref<?x?xf32>, %B: memref<?x?xf32>, %C:memref<?x?xf32>) {
  linalg.generic #matmul_trait
    ins(%A, %B : memref<?x?xf32>,
        memref<?x?xf32>)
    outs(%C : memref<?x?xf32>)
  {
  ^bb0(%a: f32, %b: f32, %c: f32) :
    %d = mulf %a, %b: f32
    %e = addf %c, %d: f32
    linalg.yield %e : f32
  }
  return
}


