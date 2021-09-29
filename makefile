.PHONY: clean
LEAN=${HOME}/work/lean4-contrib/build/stage1/bin/lean
LEANC=${HOME}/work/lean4-contrib/build/stage1/bin/leanc

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
