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
# build-mlir-air.sh <sysroot dir> <gcc version> <llvm dir> <cmakeModules dir> 
#     <mlir-aie dir> <mlir-air dir> <build dir> <install dir>
#
# e.g. build-mlir-air.sh /scratch/vck190_bare_prod_sysroot 10.2.0 /scratch/llvm 
#          /scratch/mlir-aie
#          /scratch/cmakeModules/cmakeModulesXilinx
#
# <sysroot dir>      - sysroot, absolute directory
# <gcc version>      - gcc version in sysroot (needed in many petalinux 
#                      sysroots to find imporant libs)
# <cmakeModules dir> - cmakeModules, absolute directory
# <mlir-aie dir>     - mlir-aie, absolute directory
#
# <mlir-air dir> - optional, mlir-air repo name, default is 'mlir-air'
# <build dir>    - optional, mlir-air/build dir name, default is 'build'
# <install dir>  - optional, mlir-air/install dir name, default is 'install'
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

MLIR_AIR_DIR=${6:-"mlir-air"}
BUILD_DIR=${7:-"build-aarch64"}
INSTALL_DIR=${8:-"install-aarch64"}

mkdir -p $MLIR_AIR_DIR/$BUILD_DIR
mkdir -p $MLIR_AIR_DIR/$INSTALL_DIR
cd $MLIR_AIR_DIR/$BUILD_DIR

cmake .. \
    -GNinja \
    -DAIE_DIR=${MLIR_AIE_DIR}/build-aarch64/lib/cmake/aie \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../${INSTALL_DIR} \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR} \
    -DCMAKE_SYSROOT=${CMAKE_SYSROOT} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
    -DCMAKE_USE_TOOLCHAIN_AIRHOST=TRUE \
    -DLLVM_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/mlir \
    |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
