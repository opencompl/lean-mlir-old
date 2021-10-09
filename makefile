.PHONY: all build test
all: build test

build:
	leanpkg build bin

test: build
	cd examples && PATH=../build/bin:${PATH} lit -v .
