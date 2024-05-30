#!/usr/bin/env bash

##===- utils/build-llvm-local.sh - Build LLVM on local machine --*- Script -*-===##
# 
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# 
##===----------------------------------------------------------------------===##
#
# This script build LLVM with custom options intended to be called on your
# machine where cloned llvm directory is in the current directory
#
# ./build-llvm-local.sh <llvm dir> <build dir> <install dir>
#
# <llvm dir>    - optional, default is 'llvm'
# <build dir>   - optional, default is 'build' (for llvm/build)
# <install dir> - optional, default is 'install' (for llvm/install)
#
##===----------------------------------------------------------------------===##

LLVM_DIR=${1:-"llvm"}
BUILD_DIR=${2:-"build"}
INSTALL_DIR=${3:-"install"}

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $2}'`

mkdir -p $LLVM_DIR/$BUILD_DIR
mkdir -p $LLVM_DIR/$INSTALL_DIR
LLVM_ENABLE_RTTI=${LLVM_ENABLE_RTTI:OFF}

# Enter a sub-shell to avoid messing up with current directory in case of error
(
cd $LLVM_DIR/$BUILD_DIR
set -o pipefail
set -e

CMAKE_CONFIGS="\
    -GNinja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_BUILD_UTILS=ON \
    -DLLVM_ENABLE_RTTI=$LLVM_ENABLE_RTTI \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
    -DLLVM_ENABLE_PROJECTS=clang;lld;mlir \
    -DLLVM_TARGETS_TO_BUILD:STRING=X86;ARM;AArch64; \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
    -DCLANG_LINK_CLANG_DYLIB=OFF \
    -DMLIR_BUILD_MLIR_DYLIB=OFF \
    -DLLVM_INCLUDE_UTILS=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_OPTIMIZED_TABLEGEN=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 \
    -DLLVM_DISTRIBUTION_COMPONENTS=cmake-exports;not;count;FileCheck;MLIRPythonModules;mlir-cpu-runner;mlir-linalg-ods-yaml-gen;mlir-opt;mlir-reduce;mlir-tblgen;mlir-translate;mlir-headers;mlir-cmake-exports"

if [ -x "$(command -v lld)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_USE_LINKER=lld"
fi

if [ -x "$(command -v ccache)" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_CCACHE_BUILD=ON"
fi

cmake $CMAKE_CONFIGS ../llvm 2>&1 | tee cmake.log
ninja 2>&1 | tee ninja.log
ninja install 2>&1 | tee ninja-install.log
)
