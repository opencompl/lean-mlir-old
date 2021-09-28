LEAN=${HOME}//work/lean4-contrib/build/stage1/bin/lean
LEANC=${HOME}//work/lean4-contrib/build/stage1/bin/leanc
mlir-opt:
	${LEAN} --version
	${LEAN} mlir.lean -c mlir-opt.c
	${LEANC} mlir-opt.c -o mlir-opt
