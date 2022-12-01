#!/usr/bin/env bash

##===- utils/cross-build-mlir-aie.sh - Build mlir-aie --*- Script -*-===##
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script build mlir-aie given the <sysroot dir>, <llvm dir> and
# <cmakeModules dir>. Assuming they are all in the same subfolder, it would
# look like:
#
# cross-build-mlir-aie.sh <toolchain file> <sysroot dir> <cmakeModules dir>
#     <llvm dir> <install dir> <mlir-aie dir> <build dir>
#
# <toolchain file> - absolute path to cmake toolchain file
# <sysroot dir> - sysroot, absolute directory
# <cmakeModules dir> - cmakeModules, absolute directory
# <llvm dir>     - llvm location, absolute directory
# <install dir>  - optional, default is 'install-aarch64'
# <mlir-aie dir> - optional, default is 'mlir-aie'
# <build dir>    - optional, default is '<mlir-aie dir>/build-aarch64'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 4 ]; then
    echo "ERROR: Needs at least 4 arguments for <toolchain file>, <sysroot dir>, "
    echo "<cmakeModules dir>, <llvm dir>."
    return
fi

CMAKE_TOOLCHAIN_FILE=$1
CMAKE_SYSROOT=$2
CMAKEMODULES_DIR=$3
LLVM_DIR=$4

INSTALL_DIR=${6:-"install-aarch64"}
MLIR_AIE_DIR=${7:-"mlir-aie"}
BUILD_DIR=${8:-"${MLIR_AIE_DIR}/build-aarch64"}

BUILD_DIR=`realpath ${BUILD_DIR}`
INSTALL_DIR=`realpath ${INSTALL_DIR}`

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

cmake -GNinja \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR} \
    -DCMAKE_SYSROOT=${CMAKE_SYSROOT} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
    -DLLVM_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/mlir \
    -DVitisSysroot=${CMAKE_SYSROOT} \
    .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
