#!/usr/bin/env bash
#set -x

##===- utils/build-mlir-air-pcie.sh - Build mlir-air --*- Script -*-===##
# 
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# 
##===----------------------------------------------------------------------===##
#
# This script builds mlir-air given the <llvm dir> and <mlir-aie dir>. Assuming 
# they are all in the same subfolder, it would look like:
#
# build-mlir-air-pcie.sh <llvm dir> <mlir-aie dir>
#     <build dir> <install dir>
#
# e.g. ./utils/build-mlir-air-pcie.sh 
#          utils/llvm 
#          utils/mlir-aie
#
# <llvm dir>         - llvm
# <mlir-aie dir>     - mlir-aie
#
# <build dir>    - optional, build dir name, default is 'build-pcie'
# <install dir>  - optional, install dir name, default is 'install-pcie'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 2 ]; then
    echo "ERROR: Needs at least 2 arguments for <llvm dir> and <mlir-aie dir>."
    exit 1
fi
LLVM_DIR=`realpath $1`
MLIR_AIE_DIR=`realpath $2`

CMAKEMODULES_DIR=`realpath $MLIR_AIE_DIR/cmake`

BUILD_DIR=${3:-"build-pcie"}
INSTALL_DIR=${4:-"install-pcie"}

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $2}'`

cmake .. \
    -GNinja \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/modulesXilinx \
    -Dx86_64_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86_64.cmake \
    -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
    -DAIE_DIR=${MLIR_AIE_DIR}/build/lib/cmake/aie \
    -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 \
    -DVitisSysroot="" \
    -DARM_TOOLCHAIN_OPT="" \
	-DAIR_RUNTIME_TARGETS="x86_64" \
    -DBUILD_AIR_PCIE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_USE_LINKER=lld \
    |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
