#!/usr/bin/env bash

##===- utils/build-mlir-aie.sh - Build mlir-aie --*- Script -*-===##
# 
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# 
##===----------------------------------------------------------------------===##
#
# This script build mlir-aie given the <sysroot dir>, <llvm dir> and 
# <cmakeModules dir>. Assuming they are all in the same subfolder, it would
# look like:
#
# build-mlir-aie-local.sh <llvm dir> <cmakeModules dir> 
#     <mlir-aie dir> <build dir> <install dir> <libxaie dir>
#
# e.g. build-mlir-aie-local.sh /scratch/llvm 
#     /scratch/cmakeModules/cmakeModulesXilinx
#
# <libxaie dir>  - optional, libxaie installation dir. Use "NONE" to skip.
# <mlir-aie dir> - optional, mlir-aie repo name, default is 'mlir-aie'
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs at least 2 arguments for <llvm dir> and <cmakeModules dir>."
    exit 1
fi
LLVM_DIR=`realpath $1`
CMAKEMODULES_DIR=`realpath $2`

# Set LIBXAIE_DIR only if provided and not "NONE"
if [ -n "$3" ] && [ "$3" != "NONE" ]; then
    LIBXAIE_DIR=`realpath $3`
fi
MLIR_AIE_DIR=${4:-"mlir-aie"}
BUILD_DIR=${5:-"build"}
INSTALL_DIR=${6:-"install"}

mkdir -p $MLIR_AIE_DIR/$BUILD_DIR
mkdir -p $MLIR_AIE_DIR/$INSTALL_DIR
cd $MLIR_AIE_DIR/$BUILD_DIR

CMAKE_ARGS="-GNinja"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER=clang"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER=clang++"
CMAKE_ARGS="$CMAKE_ARGS -DPython_FIND_VIRTUALENV=ONLY"
CMAKE_ARGS="$CMAKE_ARGS -DPython3_FIND_VIRTUALENV=ONLY"
CMAKE_ARGS="$CMAKE_ARGS -DLLVM_EXTERNAL_LIT=$(which lit)"
CMAKE_ARGS="$CMAKE_ARGS -DLLVM_DIR=${LLVM_DIR}/lib/cmake/llvm"
CMAKE_ARGS="$CMAKE_ARGS -DMLIR_DIR=${LLVM_DIR}/lib/cmake/mlir"

# Only add LibXAIE if LIBXAIE_DIR is set
if [ -n "$LIBXAIE_DIR" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DLibXAIE_x86_64_DIR=${LIBXAIE_DIR}"
fi

CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=../${INSTALL_DIR}"
CMAKE_ARGS="$CMAKE_ARGS -DBUILD_SHARED_LIBS=off"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release"
CMAKE_ARGS="$CMAKE_ARGS -DLLVM_ENABLE_ASSERTIONS=ON"
CMAKE_ARGS="$CMAKE_ARGS -DAIE_ENABLE_BINDINGS_PYTHON=ON"
CMAKE_ARGS="$CMAKE_ARGS -DAIE_RUNTIME_TARGETS=x86_64"
CMAKE_ARGS="$CMAKE_ARGS -DAIE_RUNTIME_TEST_TARGET=x86_64"
CMAKE_ARGS="$CMAKE_ARGS -DVITIS_VPP=$(which  v++)"
CMAKE_ARGS="$CMAKE_ARGS -DAIE_VITIS_COMPONENTS=AIE2;AIE2P"

if [ -x "$(command -v lld)" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DLLVM_USE_LINKER=lld"
fi

if [ -x "$(command -v ccache)" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DLLVM_CCACHE_BUILD=ON"
fi

cmake $CMAKE_ARGS .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
