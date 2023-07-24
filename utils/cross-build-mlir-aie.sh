#!/usr/bin/env bash

##===- utils/cross-build-mlir-aie.sh - Build mlir-aie --*- Script -*-===##
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script build mlir-aie given the <sysroot dir> and <llvm dir>. Assuming 
# they are all in the same subfolder, it would look like:
#
# cross-build-mlir-aie.sh <sysroot dir> <llvm dir> <install dir> <mlir-aie dir> 
#     <build dir>
#
# <sysroot dir> - sysroot, absolute directory
# <llvm dir>     - llvm location, absolute directory
# <install dir>  - optional, default is 'install-aarch64'
# <mlir-aie dir> - optional, default is 'mlir-aie'
# <build dir>    - optional, default is '<mlir-aie dir>/build-aarch64'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs at least 2 arguments for <sysroot dir> and <llvm dir>."
    exit 1
fi

CMAKE_SYSROOT=`realpath $1`
LLVM_DIR=`realpath $2`

INSTALL_DIR=${3:-"install-aarch64"}
MLIR_AIE_DIR=${4:-"mlir-aie"}
BUILD_DIR=${5:-"${MLIR_AIE_DIR}/build-aarch64"}

CMAKEMODULES_DIR=`realpath $MLIR_AIE_DIR/cmake`

BUILD_DIR=`realpath ${BUILD_DIR}`
INSTALL_DIR=`realpath ${INSTALL_DIR}`

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

set -o pipefail
set -e
cmake -GNinja \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/modulesXilinx \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKEMODULES_DIR}/toolchainFiles/toolchain_clang_crosscomp_pynq.cmake \
    -DSysroot=${CMAKE_SYSROOT} \
    -DArch=arm64 \
    -DLLVM_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/mlir \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -Wno-dev \
    .. |& tee cmake.log

ninja install |& tee ninja-install.log

exit 0
