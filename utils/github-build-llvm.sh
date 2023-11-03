#!/usr/bin/env bash

##===- utils/github-build-llvm.sh ---------------------------*- Script -*-===##
# 
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script build LLVM with the standard options. Intended to be called from 
# the github workflows.
#
##===----------------------------------------------------------------------===##

BUILD_DIR=${1:-"build"}
INSTALL_DIR=${2:-"install"}

mkdir -p llvm/$BUILD_DIR
mkdir -p llvm/$INSTALL_DIR
pushd llvm/$BUILD_DIR
cmake ../llvm \
  -GNinja \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DLLVM_OPTIMIZED_TABLEGEN=OFF \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_C_COMPILER=clang-12 \
  -DCMAKE_CXX_COMPILER=clang++-12 \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON

cmake --build . --target install -- -j$(nproc)
popd
