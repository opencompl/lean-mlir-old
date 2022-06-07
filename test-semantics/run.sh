#! /usr/bin/env bash

passes="--convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm"
status=0

for input in *.mlir; do
  name=${input%.mlir}
  output=${name}.llvm

  echo "> mlir-opt $input $passes | mlir-translate --mlir-to-llvmir - -o $output"
  mlir-opt $input $passes | mlir-translate --mlir-to-llvmir - -o $output
  echo "> lli $output"
  lli $output
  rc=$?

  if [[ ! $rc == 0 ]]; then
    echo -e "\e[31;1mFAIL\e[0m"
    status=1
  else
    echo -e "\e[32;1mPASS\e[0m"
  fi
done

exit $status
