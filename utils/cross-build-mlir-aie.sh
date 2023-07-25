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
# cross-build-mlir-aie.sh <toolchain fine> <sysroot dir> <llvm dir> 
#     <install dir> <mlir-aie dir> <build dir>
#
# <toolchain file> - absolute path to cmake toolchain file
# <sysroot dir> - sysroot, absolute directory
# <llvm dir>     - llvm location, absolute directory
# <install dir>  - optional, default is 'install-aarch64'
# <mlir-aie dir> - optional, default is 'mlir-aie'
# <build dir>    - optional, default is '<mlir-aie dir>/build-aarch64'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 3 ]; then
    echo "ERROR: Needs at least 3 arguments for <toolchain file>, <sysroot dir> and <llvm dir>."
    exit 1
fi

CMAKE_TOOLCHAIN_FILE=`realpath $1`
CMAKE_SYSROOT=`realpath $2`
LLVM_DIR=`realpath $3`

INSTALL_DIR=${4:-"install-aarch64"}
MLIR_AIE_DIR=${5:-"mlir-aie"}
BUILD_DIR=${6:-"${MLIR_AIE_DIR}/build-aarch64"}

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
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
    -DSysroot=${CMAKE_SYSROOT} \
    -DArch=arm64 \
    -DLLVM_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/mlir \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -Wno-dev \
    .. |& tee cmake.log

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
