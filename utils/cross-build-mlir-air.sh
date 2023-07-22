#!/usr/bin/env bash
#set -x

##===- utils/build-mlir-air.sh - Build mlir-air --*- Script -*-===##
# 
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# 
##===----------------------------------------------------------------------===##
#
# This script build mlir-air given the <sysroot dir>, <llvm dir>, 
# <cmakeModules dir>, and <mlir-aie dir>. Assuming they are all in the same
# subfolder, it would look like:
#
# build-mlir-air.sh <sysroot dir> <llvm dir> < <mlir-aie dir> <install dir> 
#     <build dir>
#
# e.g. ./utils/cross-build-mlir-air.sh 
#          /sysroot 
#          utils/llvm 
#          utils/mlir-aie 
#          install-aarch64 
#          build-aarch64
#
# <toolchain file> - path to cmake toolchain file
# <sysroot dir> - sysroot
# <cmakeModules dir> - cmakeModules
# <llvm dir>     - llvm location
# <mlir-aie dir>     - mlir-aie
# <libxaie dir>     - libxaie, default is '<cmakeModules dir>/opt/xaienginev2'
# <install dir>  - optional, default is 'install-aarch64'
# <build dir>    - optional, default is '<mlir-air dir>/build-aarch64'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 3 ]; then
    echo "ERROR: Needs at least 3 arguments for <sysroot dir>, <llvm dir>, "
    echo "and <mlir-aie dir>."
    exit 1
fi

BASE_DIR=`realpath $(dirname $0)/..`
CMAKE_TOOLCHAIN_FILE=$BASE_DIR/cmake/modules/toolchain_aarch64.cmake
CMAKE_SYSROOT=`realpath $1`
LLVM_DIR=`realpath $2`
MLIR_AIE_DIR=`realpath $3`
MLIR_AIE_CMAKEMODULES_DIR=$MLIR_AIE_DIR/cmake

INSTALL_DIR=${4:-"install-aarch64"}
BUILD_DIR=${5:-"${BASE_DIR}/build-aarch64"}

BUILD_DIR=`realpath ${BUILD_DIR}`
INSTALL_DIR=`realpath ${INSTALL_DIR}`

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

set -o pipefail

cmake .. \
    -GNinja \
    -DAIE_DIR=${MLIR_AIE_DIR}/build-aarch64/lib/cmake/aie \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DCMAKE_MODULE_PATH=${MLIR_AIE_CMAKEMODULES_DIR}/modulesXilinx \
    -DCMAKE_SYSROOT=${CMAKE_SYSROOT} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
    -DAIR_RUNTIME_TARGETS:STRING="aarch64" \
    -Daarch64_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
    -DLLVM_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/mlir \
    |& tee cmake.log

ec=$?
if [ $ec -ne 0 ]; then
    echo "CMake Error"
    exit $ec
fi

ninja install |& tee ninja-install.log

ec=$?
if [ $ec -ne 0 ]; then
    echo "CMake Error"
    exit $ec
fi
