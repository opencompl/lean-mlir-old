.PHONY: all build test debug doc
all: build test


build:
	lake build 

debug:
	lake build 2>&1 | spc -e "red, error"  -e "grn,def" # for colored output

test: build
	cd examples && PATH=../build/bin:${PATH} lit -v .

doc:
	doc-gen4 / MLIR
	cd build/doc && python3 -m http.server 80
