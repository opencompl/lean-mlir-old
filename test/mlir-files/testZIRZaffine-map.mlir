#map0 = affine_map<() -> (0)>
#map1 = affine_map<(d0, d1) -> (d0 + 1, d1 * 4 + 2)>
#map2 = affine_map<(d0, d1) -> (d1 - d0 + (d0 - d1 + 1) * 2 + d1 - 1, d1 * 4 + 2)>
#map3 = affine_map<(d0, d1) -> (d0 + 2, d1)>
#map4 = affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>
#map5 = affine_map<(d0, d1)[s0] -> (d0 + s0, d1 + 5)>
#map6 = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0, d1)>
#map7 = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0 + 5, d1)>
#map8 = affine_map<(d0, d1)[s0] -> (d0 + d1 + 5, d1)>
#map9 = affine_map<(d0, d1)[s0] -> (d0 + d1 + 5, d1)>
#map10 = affine_map<(d0, d1)[s0] -> (d0 * 2, d1 * 3)>
#map11 = affine_map<(d0, d1)[s0] -> (d0 + (d1 + s0 * 3) * 5 + 12, d1)>
#map12 = affine_map<(d0, d1)[s0] -> (d0 * 5 + d1, d1)>
#map13 = affine_map<(d0, d1)[s0] -> (d0 + d1, d1)>
#map14 = affine_map<(d0, d1)[s0] -> (d0 + d1 + 7, d1 + 3)>
#map15 = affine_map<(d0, d1)[s0] -> (d0, 0)>
#map16 = affine_map<(d0, d1)[s0] -> (d0, d1 * s0)>
#map17 = affine_map<(d0, d1) -> (d0, d0 * 3 + d1)>
#map18 = affine_map<(d0, d1) -> (d0, d0 + d1 * 3)>
#map19 = affine_map<(d0, d1)[s0] -> (d0, d0 * ((s0 * s0) * 9) + 3)>
#map20 = affine_map<(d0, d1) -> (1, d0 + d1 * 3 + 5)>
#map21 = affine_map<(d0, d1)[s0] -> (s0 * 5, d0 + d1 * 3 + d0 * 5)>
#map22 = affine_map<(d0, d1)[s0, s1] -> (d0 * (s0 * s1), d1)>
#map23 = affine_map<(d0, d1)[s0, s1] -> (d0, d1 mod 5)>
#map24 = affine_map<(d0, d1)[s0, s1] -> (d0, d1 floordiv 5)>
#map25 = affine_map<(d0, d1)[s0, s1] -> (d0, d1 ceildiv 5)>
#map26 = affine_map<(d0, d1)[s0, s1] -> (d0, d0 - d1 - 5)>
#map27 = affine_map<(d0, d1)[s0, s1] -> (d0, d0 - d1 * s1 + 2)>
#map28 = affine_map<(d0, d1)[s0, s1] -> (d0 * -5, d1 * -3, -2, -(d0 + d1), -s0)>
#map29 = affine_map<(d0, d1) -> (-4, -d0)>
#map30 = affine_map<(d0, d1)[s0, s1] -> (d0, d1 floordiv s0, d1 mod s0)>
#map31 = affine_map<(d0, d1, d2)[s0, s1, s2] -> ((d0 * s1) * s2 + d1 * s1 + d2)>
#map32 = affine_map<(d0, d1) -> (8, 4, 1, 3, 2, 4)>
#map33 = affine_map<(d0, d1) -> (4, 11, 512, 15)>
#map34 = affine_map<(d0, d1) -> (d0 * 2 + 1, d1 + 2)>
#map35 = affine_map<(d0, d1)[s0, s1] -> (d0 * s0, d0 + s0, d0 + 2, d1 * 2, s1 * 2, s0 + 2)>
#map36 = affine_map<(d0, d1)[s0] -> ((d0 * 5) floordiv 4, (d1 ceildiv 7) mod s0)>
#map37 = affine_map<(d0, d1) -> (d0 - d1 * 2, (d1 * 6) floordiv 4)>
#map38 = affine_map<(d0, d1, d2)[s0] -> (d0 + d1 + d2 + 1, d2 + d1, (d0 * s0) * 8)>
#map39 = affine_map<(d0, d1, d2) -> (0, d1, d0 * 2, 0)>
#map40 = affine_map<(d0, d1, d2) -> (d0, d0 * 4, 0, 0, 0)>
#map41 = affine_map<(d0, d1, d2) -> (d0, d0 * 4, 0, 0)>
#map42 = affine_map<(d0, d1)[s0] -> (0, 0, 0, 1)>
#map43 = affine_map<(d0, d1)[s0] -> (d0 * 2 + 1, d1 + s0)>
#map44 = affine_map<(d0) -> (-2, 1, -1)>
#map45 = affine_map<(d0) -> (d0 * 16 - (d0 + 1) + 15)>
#map46 = affine_map<(d0) -> (d0 - (d0 + 1))>
#map47 = affine_map<(d0)[s0] -> ((-s0) floordiv 4, d0 floordiv -1)>
#map48 = affine_map<() -> ()>
#map49 = affine_map<(d0, d1) -> (d0, d0 * 2 + d1 * 4 + 2, 1, 2, (d0 * 4) mod 8)>
#map50 = affine_map<(d0, d1) -> (d1, d0, 0)>
#map51 = affine_map<(d0, d1) -> (d0 * 3, (d0 + d1) * 2, d0 mod 2)>
#map52 = affine_map<(d0, d1) -> (d0 mod 5, (d1 mod 35) mod 4)>
"builtin.module"() ({
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, 1>) -> (), sym_name = "f0", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, 1>) -> (), sym_name = "f1", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<i8, #map0, 1>) -> (), sym_name = "f2", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3a", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3b", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3c", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3d", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3e", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3f", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3g", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3h", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3i", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3j", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map1, 1>) -> (), sym_name = "f3k", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map2, 1>) -> (), sym_name = "f3l", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map3, 1>) -> (), sym_name = "f4", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map4, 1>) -> (), sym_name = "f5", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map5, 1>) -> (), sym_name = "f6", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map6, 1>) -> (), sym_name = "f7", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map7, 1>) -> (), sym_name = "f8", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map8, 1>) -> (), sym_name = "f9", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map9, 1>) -> (), sym_name = "f10", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map10, 1>) -> (), sym_name = "f11", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map11, 1>) -> (), sym_name = "f12", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map12, 1>) -> (), sym_name = "f13", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map13, 1>) -> (), sym_name = "f14", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map14, 1>) -> (), sym_name = "f15", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map15, 1>) -> (), sym_name = "f16", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map16, 1>) -> (), sym_name = "f17", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map17, 1>) -> (), sym_name = "f19", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map18, 1>) -> (), sym_name = "f20", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map19, 1>) -> (), sym_name = "f18", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map20, 1>) -> (), sym_name = "f21", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map21, 1>) -> (), sym_name = "f22", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map22, 1>) -> (), sym_name = "f23", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map23, 1>) -> (), sym_name = "f24", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map24, 1>) -> (), sym_name = "f25", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map25, 1>) -> (), sym_name = "f26", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map26, 1>) -> (), sym_name = "f29", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map27, 1>) -> (), sym_name = "f30", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map28, 1>) -> (), sym_name = "f32", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map29, 1>) -> (), sym_name = "f33", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map30, 1>) -> (), sym_name = "f34", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × 4 × i8, #map31, 1>) -> (), sym_name = "f35", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map32, 1>) -> (), sym_name = "f36", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map33, 1>) -> (), sym_name = "f37", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map34, 1>) -> (), sym_name = "f38", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map35, 1>) -> (), sym_name = "f39", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map36>) -> (), sym_name = "f43", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<2 × 4 × i8, #map37>) -> (), sym_name = "f44", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × 100 × i8, #map38>) -> (), sym_name = "f45", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × 100 × i8, #map39>) -> (), sym_name = "f46", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × 100 × i8, #map40>) -> (), sym_name = "f47", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × 100 × i8, #map41>) -> (), sym_name = "f48", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × i8, #map42>) -> (), sym_name = "f49", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<100 × 100 × i8, #map43>) -> (), sym_name = "f50", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × i8, #map44>) -> (), sym_name = "f51", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × i8, #map45>) -> (), sym_name = "f52", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × i8, #map46>) -> (), sym_name = "f53", sym_visibility = "private"} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<10 × i32, #map47>) -> (), sym_name = "f54", sym_visibility = "private"} : () -> ()
  "foo.op"() {map = #map48} : () -> ()
  "func.func"() ({
  }) {function_type = (memref<1 × 1 × i8, #map49>) -> (), sym_name = "f56", sym_visibility = "private"} : () -> ()
  "f57"() {map = #map50} : () -> ()
  "f58"() {map = #map51} : () -> ()
  "f59"() {map = #map52} : () -> ()
}) : () -> ()

// -----
