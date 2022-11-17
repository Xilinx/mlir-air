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
# ./build-llvm-local.sh <toolchain file> <sysroot dir> <llvm dir> 
# <build dir> <install dir>
#
# <toolchain file> - absolute path to cmake toolchain file
# <sysroot dir> - sysroot, absolute directory 
# <gcc version> - gcc version in sysroot (needed in many petalinux sysroots to 
#                 find imporant libs)
# <cmakeModules dir> - cmakeModules, absolute directory 
#                      (usually cmakeModules/cmakeModulesXilinx)
# <llvm dir>    - optional, default is 'llvm'
# <build dir>   - optional, default is 'build-aarch64' (for llvm/build-aarch64)
# <install dir> - optional, default is 'install-aarch64' (for llvm/install-aarch64)
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs at least 2 arguments for <toolchain file>, <sysroot dir>."
    exit 1
fi

CMAKE_TOOLCHAIN_FILE=$1
CMAKE_SYSROOT=$2

LLVM_DIR=${3:-"llvm"}
BUILD_DIR=${4:-"build-aarch64"}
INSTALL_DIR=${5:-"install-aarch64"}

mkdir -p $LLVM_DIR/$BUILD_DIR
mkdir -p $LLVM_DIR/$INSTALL_DIR
cd $LLVM_DIR/$BUILD_DIR

cmake ../${LLVM_DIR} \
  -GNinja \
  -DCLANG_LINK_CLANG_DYLIB=OFF \
  -DCMAKE_SYSROOT=${CMAKE_SYSROOT} \
  -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
  -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
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
  -DPython3_EXECUTABLE=/usr/bin/python3.8 \
  |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
