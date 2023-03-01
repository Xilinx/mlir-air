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
# build-mlir-air.sh <toolchain file> <sysroot dir> <cmakeModules dir>
#     <llvm dir> < <mlir-aie dir> <install dir> <mlir-air dir> <build dir>
#
# e.g. build-mlir-air.sh /scratch/vck190_bare_prod_sysroot 10.2.0 /scratch/llvm 
#          /scratch/mlir-aie
#          /scratch/cmakeModules/cmakeModulesXilinx
#
# <toolchain file> - absolute path to cmake toolchain file
# <sysroot dir> - sysroot, absolute directory
# <cmakeModules dir> - cmakeModules, absolute directory
# <llvm dir>     - llvm location, absolute directory
# <mlir-aie dir>     - mlir-aie, absolute directory
# <install dir>  - optional, default is 'install-aarch64'
# <mlir-air dir> - optional, default is 'mlir-air'
# <build dir>    - optional, default is '<mlir-air dir>/build-aarch64'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 5 ]; then
    echo "ERROR: Needs at least 5 arguments for <sysroot dir>, <llvm dir>, "
    echo "<cmakeModules dir> and <mlir-aie dir>."
    exit 1
fi

CMAKE_TOOLCHAIN_FILE=$1
CMAKE_SYSROOT=$2
CMAKEMODULES_DIR=$3
LLVM_DIR=$4
MLIR_AIE_DIR=$5

INSTALL_DIR=${6:-"install-aarch64"}
MLIR_AIR_DIR=${7:-"mlir-air"}
BUILD_DIR=${8:-"${MLIR_AIR_DIR}/build-aarch64"}

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
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR} \
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