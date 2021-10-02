.PHONY: clean foo
LEAN=${HOME}/work/lean4-contrib/build/stage1/bin/lean
LEANC=${HOME}/work/lean4-contrib/build/stage1/bin/leanc

LEAN_PATH=/home/bollu/work/2-mlir-verif

foo:
	LEAN_PATH=${LEAN_PATH} && lean -o MLIR/Doc.olean MLIR/Doc.lean
	LEAN_PATH=${LEAN_PATH} && lean -o MLIR/AST.olean MLIR/AST.lean
	LEAN_PATH=${LEAN_PATH} lean Foo.lean

mlir-opt: mlir.lean
	@${LEAN} --version
	${LEAN} mlir.lean -c mlir-opt.c
	${LEANC} mlir-opt.c -o mlir-opt

# useful to debug crashes
crash: crash.lean
	${LEAN} --version
	${LEAN} crash.lean -c crash.c
	${LEANC} crash.c -o crash

clean:
	-rm mlir-opt.c
	-rm mlir-opt
