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
# build-mlir-aie.sh <sysroot dir> <llvm dir> <cmakeModules dir> 
#     <mlir-aie dir> <build dir> <install dir>
#
# e.g. build-mlir-aie.sh /scratch/vck190_bare_prod_sysroot /scratch/llvm 
#          /scratch/cmakeModules/cmakeModulesXilinx
#
# <mlir-aie dir> - optional, mlir-aie repo name, default is 'mlri-aie'
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 3 ]; then
    echo "ERROR: Needs at least 3 arguments for <sysroot dir>, <llvm dir> and <cmakeModules dir>."
    return
fi
SYSROOT_DIR=$1
LLVM_DIR=$2
CMAKEMODULES_DIR=$3

MLIR_AIE_DIR=${4:-"mlir-aie"}
BUILD_DIR=${5:-"build"}
INSTALL_DIR=${6:-"install"}

mkdir -p $MLIR_AIE_DIR/$BUILD_DIR
mkdir -p $MLIR_AIE_DIR/$INSTALL_DIR
cd $MLIR_AIE_DIR/$BUILD_DIR
cmake -GNinja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/ \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DVitisSysroot=${SYSROOT_DIR} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_SHARED_LIBS=off \
    .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
ninja check-aie |& tee ninja-check-aie.log
