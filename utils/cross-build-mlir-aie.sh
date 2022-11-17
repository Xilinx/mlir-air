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
#     <llvm dir> <mlir-aie dir> <build dir> <install dir>
#
# <toolchain file> - absolute path to cmake toolchain file
# <sysroot dir> - sysroot, absolute directory 
# <cmakeModules dir> - cmakeModules, absolute directory 
#                      (usually cmakeModules/cmakeModulesXilinx)
# <llvm dir>    - llvm location, absolute directory
# <mlir-aie dir> - optional, mlir-aie repo name, default is 'mlir-aie'
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 4 ]; then
    echo "ERROR: Needs at least 4 arguments for <toolchain file>, <sysroot dir>, "
    echo "<cmakeModules dir>, <llvm dir>."
    exit 1
fi

CMAKE_TOOLCHAIN_FILE=$1
CMAKE_SYSROOT=$2
CMAKEMODULES_DIR=$3
LLVM_DIR=$4

MLIR_AIE_DIR=${5:-"mlir-aie"}
BUILD_DIR=${6:-"build-aarch64"}
INSTALL_DIR=${7:-"install-aarch64"}

mkdir -p $MLIR_AIE_DIR/$BUILD_DIR
mkdir -p $MLIR_AIE_DIR/$INSTALL_DIR
cd $MLIR_AIE_DIR/$BUILD_DIR

cmake -GNinja \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../${INSTALL_DIR} \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR} \
    -DCMAKE_SYSROOT=${CMAKE_SYSROOT} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
    -DLLVM_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build-aarch64/lib/cmake/mlir \
    -DVitisSysroot=${CMAKE_SYSROOT} \
    .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
