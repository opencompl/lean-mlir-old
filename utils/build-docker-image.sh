#!/usr/bin/env sh
# Builds the docker build image which is used by our CI.
set -e
set -o xtrace
docker build . -t opencompl/lean-mlir
docker push opencompl/lean-mlir

