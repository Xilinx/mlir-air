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
# ./build-llvm-local.sh <toolchain file> <sysroot dir> <install dir> <llvm dir>
# <build dir>
#
# <toolchain file> - absolute path to cmake toolchain file
# <sysroot dir> - sysroot, absolute directory
# <install dir> - optional, default is 'install-aarch64'
# <llvm dir>    - optional, default is 'llvm'
# <build dir>   - optional, default is '<llvm dir>/build-aarch64'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs at least 2 arguments for <toolchain file>, <sysroot dir>."
    exit 1
fi

CMAKE_TOOLCHAIN_FILE=`realpath $1`
CMAKE_SYSROOT=`realpath $2`

INSTALL_DIR=${3:-"install-aarch64"}
LLVM_DIR=${4:-"llvm"}
BUILD_DIR=${5:-"${LLVM_DIR}/build-aarch64"}

BUILD_DIR=`realpath ${BUILD_DIR}`
INSTALL_DIR=`realpath ${INSTALL_DIR}`

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

set -o pipefail

cmake ../llvm \
  -GNinja \
  -DCLANG_LINK_CLANG_DYLIB=OFF \
  -DCMAKE_SYSROOT=${CMAKE_SYSROOT} \
  -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DMLIR_BUILD_MLIR_DYLIB=OFF \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_DEFAULT_TARGET_TRIPLE="aarch64-linux-gnu" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_INCLUDE_UTILS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DLLVM_OPTIMIZED_TABLEGEN=OFF \
  -DLLVM_TARGET_ARCH="AArch64" \
  -DLLVM_TARGETS_TO_BUILD:STRING="ARM;AArch64" \
  |& tee cmake.log

ec=$?
if [ $ec -ne 0 ]; then
    echo "CMake Error"
    exit $ec
fi

ninja install |& tee ninja-install.log

ec=$?
if [ $ec -ne 0 ]; then
    echo "Build Error"
    exit $ec
fi

exit 0
