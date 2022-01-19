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



//  #map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
//  #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
//  #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
//  "module"() ( {
//    "func"() ( {
//    ^bb0(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>):  // no predecessors
//      "linalg.generic"(%arg0, %arg1, %arg2) ( {
//      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
//        %0 = "std.mulf"(%arg3, %arg4) : (f32, f32) -> f32
//        %1 = "std.addf"(%arg5, %0) : (f32, f32) -> f32
//        "linalg.yield"(%1) : (f32) -> ()
//      }) {doc = "C(m, n) += A(m, k) * B(k, n)", indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], library_call = "linalg_matmul", operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
//      "std.return"() : () -> ()
//    }) {sym_name = "main", type = (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()} : () -> ()
//    "module_terminator"() : () -> ()
//  }) : () -> ()

