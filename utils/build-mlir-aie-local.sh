#!/usr/bin/env bash

##===- utils/build-mlir-aie.sh - Build mlir-aie --*- Script -*-===##
# 
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# 
##===----------------------------------------------------------------------===##
#
# This script build mlir-aie given the <llvm dir> and <libxaie dir>. Assuming 
# they are all in the same subfolder, it would look like:
#
# build-mlir-aie-local.sh <llvm dir> <libxaie dir> <mlir-aie dir> <build dir> 
#     <install dir>
#
# e.g. build-mlir-aie-local.sh /scratch/llvm /opt/xaiengine/
#
# <libxaie dir>  - libxaie installation dir, default is '/opt/xaiengine'
# <mlir-aie dir> - optional, mlir-aie repo name, default is 'mlir-aie'
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs at least 2 arguments for <llvm dir> and <libxaie dir>."
    exit 1
fi

LLVM_DIR=`realpath $1`
LibXAIE_x86_64_DIR=`realpath ${2:-"/opt/xaiengine/"}`

MLIR_AIE_DIR=${3:-"mlir-aie"}
BUILD_DIR=${4:-"build"}
INSTALL_DIR=${5:-"install"}

CMAKEMODULES_DIR=`realpath $MLIR_AIE_DIR/cmake`

mkdir -p $MLIR_AIE_DIR/$BUILD_DIR
mkdir -p $MLIR_AIE_DIR/$INSTALL_DIR
cd $MLIR_AIE_DIR/$BUILD_DIR
set -o pipefail
set -e

cmake -GNinja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
    -DLibXAIE_x86_64_DIR=${LibXAIE_x86_64_DIR} \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/modulesXilinx \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    "-DAIE_RUNTIME_TARGETS=x86_64;aarch64" \
    -DAIE_RUNTIME_TEST_TARGET=aarch64 \
    .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
