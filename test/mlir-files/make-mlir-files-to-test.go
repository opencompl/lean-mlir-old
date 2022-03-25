package main

import (
	"bytes"
	"fmt" // A package in the Go standard library.
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func mk_testfile_contents(mlir_contents string, print_path string) string {
	out := fmt.Sprintf(`
import MLIR.Doc
import MLIR.AST
import MLIR.EDSL

open  MLIR.EDSL
open MLIR.AST
open MLIR.Doc
open IO

-- | write an op into the path
def o: Op := [mlir_op|
%s
] 
-- | main program
def main : IO Unit :=
    let str := Pretty.doc o
    FS.writeFile "%s" str
`, mlir_contents, print_path)
	return out
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func fileNameWithoutExtTrimSuffix(fileName string) string {
	return strings.TrimSuffix(fileName, filepath.Ext(fileName))
}

func makeOutPath(rootPath, testFileDir, testFileNameWithoutExtension string) string {
	testFilePathWithoutRoot := strings.TrimPrefix(testFileDir, rootPath)
	testFilePathAts := strings.Replace(testFilePathWithoutRoot, "/", "Z", -1)
	return testFilePathAts + testFileNameWithoutExtension + ".mlir"
}

func main() {
	MLIR_ROOT_PATH := "/home/siddu_druid/work/llvm-project/mlir/"
	MLIR_GLOB_PATH := "/home/siddu_druid/work/llvm-project/mlir/test/*/*.mlir"
	log.Output(0, fmt.Sprintf("globbing from %s", MLIR_GLOB_PATH))

	testfiles, err := filepath.Glob(MLIR_GLOB_PATH)
	check(err)

	log.Output(0, fmt.Sprintf("globbed %d files.", len(testfiles)))
	for i, testFilePath := range testfiles {
		log.Output(1, fmt.Sprintf("%d: %s", i, testFilePath))
	}

	numSuccesses := 0
	numFailures := 0
	failureFiles := []string{}
	for _, testFilePath := range testfiles {
		testFileDir, testFileNameWithExtension := filepath.Split(testFilePath)
		testFileNameWithoutExtension := fileNameWithoutExtTrimSuffix(testFileNameWithExtension)

		// --- mlir opt canonicalization ---
		canonCmd := exec.Command("mlir-opt", testFilePath, "--mlir-print-op-generic", "--allow-unregistered-dialect", "--split-input-file")
		check(err)
		outPath := makeOutPath(MLIR_ROOT_PATH, testFileDir, testFileNameWithoutExtension)
		canonStdout, err := os.Create(outPath)
		check(err)

		canonCmd.Stdout = canonStdout // write output into canon out file.
		var canonStderr bytes.Buffer
		canonCmd.Stderr = &canonStderr
		check(err)
		log.Output(0, fmt.Sprintf("Running | %s |.", canonCmd.String()))
		err = canonCmd.Run()
		if err == nil {
			numSuccesses++
			canonStdout.Close()
		} else {
			failureFiles = append(failureFiles, testFilePath)
			numFailures++
			log.Output(0, fmt.Sprintf("Error | %s |.", canonStderr.String()))
			canonStdout.Close()
			os.Remove(outPath)
		}
	}

	log.Output(0, fmt.Sprintf("total: %d | success: %d  | failure: %d", numSuccesses+numFailures, numSuccesses, numFailures))
	for i, failureFile := range failureFiles {
		log.Output(0, fmt.Sprintf("failure %4d: %s", i+1, failureFile))
	}
}
