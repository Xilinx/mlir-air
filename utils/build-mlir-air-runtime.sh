#!/usr/bin/env bash
#set -x

##===- utils/build-mlir-air-runtime.sh - Build mlir-air --*- Script -*-===##
# 
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# 
##===----------------------------------------------------------------------===##
#
# This script build mlir-air (with air runtime) given the <sysroot dir>, 
# <llvm dir>, <cmakeModules dir>, <mlir-aie dir>, <mlir-air dir> and 
# <libxaie dir>. Assuming they are all in the same subfolder, it would look 
# like:
#
# build-mlir-air.sh <sysroot dir> <llvm dir> <cmakeModules dir> <mlir-aie dir>
#      <mlir-air dir> <libxaie dir> <build dir> <install dir>
#
# e.g. build-mlir-air.sh /scratch/vck190_air_sysroot /scratch/llvm 
#          /scratch/cmakeModules
#          /scratch/mlir-aie /scratch/mlir-air /scratch/libxaie_install
#
# <sysroot dir>      - sysroot
# <llvm dir>         - llvm
# <cmakeModules dir> - cmakeModules
# <mlir-aie dir>     - mlir-aie
# <mlir-air dir>     - mlir-air
# <libxaie dir>      - libxaie-install
#
# <build dir>    - optional, build dir name, default is 'build'
# <install dir>  - optional, install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 6 ]; then
    echo "ERROR: Needs at least 6 arguments for <sysroot dir>, <llvm dir>,"
    echo "<cmakeModules dir>, <mlir-aie dir>, <mlir-air dir> and "
    echo "<libxaie-install dir>."
    exit 1
fi
SYSROOT_DIR=`realpath $1`
LLVM_DIR=`realpath $2`
CMAKEMODULES_DIR=`realpath $3`
MLIR_AIE_DIR=`realpath $4`
MLIR_AIR_DIR=`realpath $5`
LIBXAIE_INSTALL=`realpath $6`

BUILD_DIR=${7:-"build"}
INSTALL_DIR=${8:-"install"}

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $2}'`

cmake .. \
    -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=${MLIR_AIR_DIR}/cmake/modules/toolchain_x86.cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/cmakeModulesXilinx \
    -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
    -DAIE_DIR=${MLIR_AIE_DIR}/build/lib/cmake/aie \
    -DLibXAIE_ROOT=${LIBXAIE_INSTALL} \
    -DAIR_RUNTIME_TARGETS:STRING="x86" \
    -Dx86_TOOLCHAIN_FILE=${MLIR_AIR_DIR}/cmake/modules/toolchain_x86.cmake \
    -DLLVM_EXTERNAL_LIT=${LLVM_DIR}/build/bin/llvm-lit \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX=install \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
