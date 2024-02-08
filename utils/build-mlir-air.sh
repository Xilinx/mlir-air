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
# build-mlir-air.sh <sysroot dir> <llvm dir> <cmakeModules dir> <mlir-aie dir>
#     <build dir> <install dir>
#
# e.g. build-mlir-air.sh /scratch/vck190_air_sysroot /scratch/llvm 
#          /scratch/cmakeModules/cmakeModulesXilinx /scratch/mlir-aie build install
#
# <sysroot dir>      - sysroot
# <llvm dir>         - llvm
# <cmakeModules dir> - cmakeModules
# <mlir-aie dir>     - mlir-aie
#
# <build dir>    - optional, build dir name, default is 'build'
# <install dir>  - optional, install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 4 ]; then
    echo "ERROR: Needs at least 4 arguments for <sysroot dir>, <llvm dir>,"
    echo "<cmakeModules dir> and <mlir-aie dir>."
    exit 1
fi
SYSROOT_DIR=`realpath $1`
LLVM_DIR=`realpath $2`
CMAKEMODULES_DIR=`realpath $3`
MLIR_AIE_DIR=`realpath $4`

BUILD_DIR=${5:-"build"}
INSTALL_DIR=${6:-"install"}

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $2}'`

cmake .. \
    -GNinja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DArch=arm64 \
    -DgccVer=10.2.0 \
    -DCMAKE_USE_TOOLCHAIN=FALSE \
    -DCMAKE_USE_TOOLCHAIN_AIRHOST=TRUE \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
    -DAIE_DIR=${MLIR_AIE_DIR}/build/lib/cmake/aie \
    -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/ \
    |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
