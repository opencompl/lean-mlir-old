package main

import (
	"flag"
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

open Lean
open Lean.Parser
open  MLIR.EDSL
open MLIR.AST
open MLIR.Doc
open IO

set_option maxHeartbeats 999999999


declare_syntax_cat mlir_ops
syntax (ws mlir_op ws)* : mlir_ops
syntax "[mlir_ops|" mlir_ops "]" : term

macro_rules
|`+"`"+`([mlir_ops| $[ $xs ]* ]) => do 
  let xs <- xs.mapM (fun x =>`+"`"+`([mlir_op| $x]))
  quoteMList xs.toList

  
-- | write an op into the path
def o: List Op := [mlir_ops|
%s
] 
-- | main program
def main : IO Unit :=
    let str := Doc.VGroup (o.map Pretty.doc)
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

var FlagStopOnCompileError bool = false

func main() {
	flag.BoolVar(&FlagStopOnCompileError, "stop-on-compile-error", false, "Enable if script should stop when Lean build fails")
	flag.Parse()
	MLIR_GLOB_PATH := "./mlir-files/*.mlir"
	log.Output(0, fmt.Sprintf("globbing from %s", MLIR_GLOB_PATH))

	testfiles, err := filepath.Glob(MLIR_GLOB_PATH)
	check(err)

	log.Output(0, fmt.Sprintf("globbed %d files.", len(testfiles)))
	for i, testFilePath := range testfiles {
		log.Output(1, fmt.Sprintf("%d: %s", i, testFilePath))
	}

	successFiles := []string{}
	failureFiles := []string{}

	for iprogress, testFilePath := range testfiles {
		fmt.Printf("PROGRESS %4d/%4d: %4.2f | %%", iprogress, len(testfiles),
			float32(iprogress)/float32(len(testfiles))*100.)
		successRatio := float32(len(successFiles)) / float32(len(testfiles))
		fmt.Printf(" | num success: %4d | num failures: %4d  | success ratio: %4.2f\n",
			len(successFiles), len(failureFiles), successRatio*100.)

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
			log.Output(0, fmt.Sprintf("Leanc error out: | %s |", buildCmdOut))
			failureFiles = append(failureFiles, testFilePath)

			if FlagStopOnCompileError {
				panic(err)
			} else {
				continue
			}
		}

		// ---- Run project
		runCmd := exec.Command("./build/bin/TestCanonicalizer")
		log.Output(0, fmt.Sprintf("Running | %s |.", runCmd.String()))
		runCmdOut, err := runCmd.CombinedOutput()
		if err != nil {
			log.Output(0, fmt.Sprintf("Failure when running!: | %s |", runCmdOut))
			panic(err)
		}

		// --- write canonicalized file output
		canonOutFilePath := filepath.Join(testFileDir, testFileNameWithoutExtension, ".canon")
		ioutil.WriteFile(canonOutFilePath, inputContents, 0666)

		// --- run diff
		// TODO: figure out how to write command line tools in go.
		/*
			diffCmd := exec.Command("diff", canonOutFilePath, leanFileStdoutPath)
			diffCmdOut, err := diffCmd.CombinedOutput()
			if err != nil {
				log.Output(0, fmt.Sprintf("diff error out: | %s |", diffCmdOut))
				panic(err)
			}
		*/
		successFiles = append(successFiles, testFilePath)
	}

	totalFiles := len(successFiles) + len(failureFiles)
	successRatio := float32(len(successFiles)) / float32(totalFiles)
	log.Output(0, fmt.Sprintf("num success: %4d | num failures: %4d | total: %4d | success ratio: %4.2f",
		len(successFiles), len(failureFiles), totalFiles, successRatio*100.))

}
