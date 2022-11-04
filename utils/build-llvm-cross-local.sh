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
# ./build-llvm-local.sh <sysroot dir> <gcc version> <cmakeModules dir> <llvm dir> 
# <build dir> <install dir>
#
# <sysroot dir> - sysroot, absolute directory 
# <gcc version> - gcc version in sysroot (needed in many petalinux sysroots to 
#                 find imporant libs)
# <cmakeModules dir> - cmakeModules, absolute directory 
#                      (usually cmakeModules/cmakeModulesXilinx)
# <llvm dir>    - optional, default is 'llvm'
# <build dir>   - optional, default is 'build' (for llvm/build)
# <install dir> - optional, default is 'install' (for llvm/install)
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 4 ]; then
    echo "ERROR: Needs at least 3 arguments for <sysroot dir>, <gcc version>, "
    echo "<cmakeModules dir>."
    exit 1
fi

SYSROOT_DIR=$1
GCC_VER=$2
CMAKEMODULES_DIR=$3
LLVM_DIR=${4:-"llvm"}
BUILD_DIR=${5:-"build"}
INSTALL_DIR=${6:-"install"}

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $2}'`

mkdir -p $LLVM_DIR/$BUILD_DIR
mkdir -p $LLVM_DIR/$INSTALL_DIR
cd $LLVM_DIR/$BUILD_DIR


cmake ../${LLVM_DIR} \
  -GNinja \
  -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR} \
  -DCMAKE_TOOLCHAIN_FILE=${CMAKEMODULES_DIR}/toolchain_clang_crosscomp_arm_petalinux.cmake \
  -DArch=arm64 \
  -DgccVer=${GCC_VER} \
  -DSysroot=${SYSROOT_DIR} \
  -DPython3_FIND_VIRTUALENV=ONLY \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_USE_LINKER=lld \
  -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DLLVM_TARGETS_TO_BUILD:STRING="X86;ARM;AArch64" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DCLANG_LINK_CLANG_DYLIB=OFF \
  -DMLIR_BINDINGS_PYTHON_ENABLED=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_BUILD_MLIR_DYLIB=OFF \
  -DLLVM_INCLUDE_UTILS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF \
  -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 \
  -DLLVM_DISTRIBUTION_COMPONENTS="cmake-exports;not;count;FileCheck;MLIRPythonModules;mlir-linalg-ods-yaml-gen;mlir-opt;mlir-reduce;mlir-tblgen;mlir-translate;mlir-headers;mlir-cmake-exports" \
  -DLLVM_DEFAULT_TARGET_TRIPLE="aarch64-linux-gnu" \
  -DLLVM_TARGET_ARCH="AArch64" \
  |& tee cmake.log

#  -DLLVM_DISTRIBUTION_COMPONENTS="mlir-cpu-runner;"
#  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \ # causes numpy error

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
