.PHONY: all build test debug
all: build test


build:
	leanpkg build bin

debug:
	leanpkg build bin 2>&1 | spc -e "red, error" # for colored output

test: build
	cd examples && PATH=../build/bin:${PATH} lit -v .
