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
		outPath := makeOutPath(MLIR_ROOT_PATH, testFileDir, testFileNameWithoutExtension)

		// --- mlir opt canonicalization ---
		{
			canonCmd := exec.Command("mlir-opt", testFilePath, "--mlir-print-op-generic", "--allow-unregistered-dialect", "--split-input-file")
			check(err)
			canonStdout, err := os.Create(outPath)
			check(err)

			canonCmd.Stdout = canonStdout // write output into canon out file.
			var canonStderr bytes.Buffer
			canonCmd.Stderr = &canonStderr
			check(err)
			log.Output(0, fmt.Sprintf("Running | %s |.", canonCmd.String()))
			err = canonCmd.Run()
			if err != nil {
				failureFiles = append(failureFiles, testFilePath)
				numFailures++
				log.Output(0, fmt.Sprintf("Error | %s |.", canonStderr.String()))
				canonStdout.Close()
				os.Remove(outPath)
				continue
			}
			numSuccesses++
			canonStdout.Close()
		} // end run canonicalizataion block

		{
			// --- run sed to replace memref<blahxblah> with memref<blah \times blah>
			// [^blah]: negated capture group for blah
			// sedCommand := exec.Command("sed", "-i", `s/<([^x]*)x([^>]*)>/<\1 × \2>/g`, outPath)
			// sedCommand := exec.Command("sed", "-i", `s/<([^x]*)x([^>]*)>/<\1 BAR \2>/g`, outPath)
			sedCommand := exec.Command("sed", "-i", "-r", `s/<([^x>]*)x([^x>]*)>/<\1 × \2>/g`, outPath)

			log.Output(0, fmt.Sprintf("Running | %s |.", sedCommand.String()))
			var sedStderr bytes.Buffer
			sedCommand.Stderr = &sedStderr
			err = sedCommand.Run()
			if err != nil {
				log.Output(0, fmt.Sprintf("Error | %s |.", sedStderr.String()))
			}
			check(err)
		} // end sed run block

		{
			// --- run sed to replace memref<blahxblah> with memref<blah \times blah>
			// [^blah]: negated capture group for blah
			// sedCommand := exec.Command("sed", "-i", `s/<([^x]*)x([^>]*)>/<\1 × \2>/g`, outPath)
			// sedCommand := exec.Command("sed", "-i", `s/<([^x]*)x([^>]*)>/<\1 BAR \2>/g`, outPath)
			sedCommand := exec.Command("sed", "-i", "-r", `s/<([^x>]*)x([^x>]*)x([^x>]*)>/<\1 × \2 × \3>/g`, outPath)

			log.Output(0, fmt.Sprintf("Running | %s |.", sedCommand.String()))
			var sedStderr bytes.Buffer
			sedCommand.Stderr = &sedStderr
			err = sedCommand.Run()
			if err != nil {
				log.Output(0, fmt.Sprintf("Error | %s |.", sedStderr.String()))
			}
			check(err)
		} // end sed run block

		{
			// --- run sed to replace memref<blahxblah> with memref<blah \times blah>
			// [^blah]: negated capture group for blah
			// sedCommand := exec.Command("sed", "-i", `s/<([^x]*)x([^>]*)>/<\1 × \2>/g`, outPath)
			// sedCommand := exec.Command("sed", "-i", `s/<([^x]*)x([^>]*)>/<\1 BAR \2>/g`, outPath)
			sedCommand := exec.Command("sed", "-i", "-r", `s/<([^x>]*)x([^x>]*)x([^x>]*)x([^x>]*)>/<\1 × \2 × \3 × \4>/g`, outPath)

			log.Output(0, fmt.Sprintf("Running | %s |.", sedCommand.String()))
			var sedStderr bytes.Buffer
			sedCommand.Stderr = &sedStderr
			err = sedCommand.Run()
			if err != nil {
				log.Output(0, fmt.Sprintf("Error | %s |.", sedStderr.String()))
			}
			check(err)
		} // end sed run block

		{
			// --- run sed to remove comments
			sedCommand := exec.Command("sed", "-i", "-r", `s-//.*--g`, outPath)

			log.Output(0, fmt.Sprintf("Running | %s |.", sedCommand.String()))
			var sedStderr bytes.Buffer
			sedCommand.Stderr = &sedStderr
			err = sedCommand.Run()
			if err != nil {
				log.Output(0, fmt.Sprintf("Error | %s |.", sedStderr.String()))
			}
			check(err)
		} // end sed run block

	}

	log.Output(0, fmt.Sprintf("total: %d | success: %d  | failure: %d", numSuccesses+numFailures, numSuccesses, numFailures))
	for i, failureFile := range failureFiles {
		log.Output(0, fmt.Sprintf("failure %4d: %s", i+1, failureFile))
	}
}
