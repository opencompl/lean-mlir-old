no-attributes-total:
	cd mlir-files && go run make-mlir-files-to-test.go  -remove-attributes-and-types
	go run run-mlir-round-trip-tests.go 

no-attributes-debugging:
	cd mlir-files && go run make-mlir-files-to-test.go  -remove-attributes-and-types
	go run run-mlir-round-trip-tests.go -stop-on-compile-error



debugging:
	cd mlir-files && go run make-mlir-files-to-test.go  -
	go run run-mlir-round-trip-tests.go -stop-on-compile-error

total:
	make -C mlir-files/
	go run run-mlir-round-trip-tests.go
