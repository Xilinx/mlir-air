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
# <mlir-aie dir> - optional, mlir-aie repo name, default is 'mlir-aie'
# <build dir>    - optional, mlir-aie/build dir name, default is 'build'
# <install dir>  - optional, mlir-aie/install dir name, default is 'install'
# <libxaie dir>  - optional, libxaie installation dir, default is 
#     '/opt/xaiengine'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs at least 2 arguments for <llvm dir> and <cmakeModules dir>."
    exit 1
fi
LLVM_DIR=`realpath $1`
CMAKEMODULES_DIR=`realpath $2`

LIBXAIE_DIR=`realpath ${3:-"/opt/xaiengine/"}`
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
    -DLibXAIE_x86_64_DIR=${LIBXAIE_DIR} \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/ \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DBUILD_SHARED_LIBS=off \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DAIE_ENABLE_BINDINGS_PYTHON=ON \
    .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
