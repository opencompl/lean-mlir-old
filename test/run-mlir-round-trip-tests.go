package main

import (
	"fmt" // A package in the Go standard library.
	"io/ioutil"
	"log"
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

func main() {
	MLIR_GLOB_PATH := "./mlir-files/*.mlir"
	log.Output(0, fmt.Sprintf("globbing from %s", MLIR_GLOB_PATH))

	testfiles, err := filepath.Glob(MLIR_GLOB_PATH)
	check(err)

	log.Output(0, fmt.Sprintf("globbed %d files.", len(testfiles)))
	for i, testFilePath := range testfiles {
		log.Output(1, fmt.Sprintf("%d: %s", i, testFilePath))
	}

	for _, testFilePath := range testfiles {
		testFileDir, testFileNameWithExtension := filepath.Split(testFilePath)
		testFileNameWithoutExtension := fileNameWithoutExtTrimSuffix(testFileNameWithExtension)
		// --- open test file
		log.Output(0, fmt.Sprintf("Reading | %s |.", testFilePath))
		inputContents, err := ioutil.ReadFile(testFilePath)
		if err != nil {
			log.Output(0, fmt.Sprintf("Error opening test file: | %s |.", err.Error()))
			panic(err)
		}
		// --- run lean through round trip ---
		// ----- Write lean
		const leanFilePath = "TestCanonicalizer.lean"
		leanFileStdoutPath := testFileNameWithoutExtension + ".out.txt"
		leanFileContents := mk_testfile_contents(string(inputContents), leanFileStdoutPath)
		log.Output(0, fmt.Sprintf("Writing | %s |.", leanFilePath))
		err = ioutil.WriteFile(leanFilePath, []byte(leanFileContents), 0666)
		check(err)

		// ---- Compile project
		buildCmd := exec.Command("lake", "build")
		log.Output(0, fmt.Sprintf("Compiling | %s |.", buildCmd.String()))
		buildCmdOut, err := buildCmd.CombinedOutput()
		if err != nil {
			log.Output(0, fmt.Sprintf("Leanc out: | %s |", buildCmdOut))
			panic(err)
		}

		// ---- Run project
		runCmd := exec.Command("./build/bin/TestCanonicalizer")
		log.Output(0, fmt.Sprintf("Running | %s |.", runCmd.String()))
		runCmdOut, err := runCmd.CombinedOutput()
		if err != nil {
			log.Output(0, fmt.Sprintf("Leanc out: | %s |", runCmdOut))
			panic(err)
		}

		// --- write canonicalized file output
		canonOutFilePath := filepath.Join(testFileDir, testFileNameWithoutExtension, ".canon")
		ioutil.WriteFile(canonOutFilePath, inputContents, 0666)

		// --- run diff
		diffCmd := exec.Command("diff", canonOutFilePath, leanFileStdoutPath)
		err = diffCmd.Run()
		check(err)
	}

}
