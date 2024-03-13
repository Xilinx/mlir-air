#!/usr/bin/env bash
#set -x

##===- utils/build-mlir-air-xrt.sh - Build mlir-air --*- Script -*-===##
# 
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# 
##===----------------------------------------------------------------------===##
#
# This script builds mlir-air given the <llvm dir>, <cmakeModules dir>,
# <mlir-aie dir> <libxaie dir> and <xrt dir>. Assuming they are all in the same 
# subfolder, it would look like:
#
# build-mlir-air-xrt.sh <llvm dir> <cmakeModules dir> <mlir-aie dir>
#     <xrt dir> <build dir> <install dir>
#
# e.g. build-mlir-air-xrt.sh /scratch/llvm 
#          /scratch/cmakeModules/cmakeModulesXilinx /scratch/mlir-aie 
#          /scratch/libxaie/install /scratch/xrt
#
# <llvm dir>         - llvm
# <cmakeModules dir> - cmakeModules
# <mlir-aie dir>     - mlir-aie
#
# <libXAIE dir>      - libXAIE to build runtime
# <XRT dir>          - XRT build runtime
#
# <build dir>    - optional, build dir name, default is 'build-xrt'
# <install dir>  - optional, install dir name, default is 'install-xrt'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 3 ]; then
    echo "ERROR: Needs at least 3 arguments for <llvm dir>, <cmakeModules dir> and <mlir-aie dir>."
    exit 1
fi
LLVM_DIR=`realpath $1`
CMAKEMODULES_DIR=`realpath $2`
MLIR_AIE_DIR=`realpath $3`

LibXAIE_DIR=`realpath $4`

XRT_DIR=`realpath $5`

BUILD_DIR=${6:-"build-xrt"}
INSTALL_DIR=${7:-"install-xrt"}

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $2}'`

cmake .. \
    -GNinja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/ \
    -DCMAKE_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86_64.cmake \
    -Dx86_64_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86_64.cmake \
    -DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
    -DAIE_DIR=${MLIR_AIE_DIR}/build/lib/cmake/aie \
    -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 \
    -DVitisSysroot="" \
    -DLibXAIE_ROOT=${LibXAIE_DIR} \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DXRT_LIB_DIR=${XRT_DIR}/lib \
    -DXRT_BIN_DIR=${XRT_DIR}/bin \
    -DXRT_INCLUDE_DIR=${XRT_DIR}/include \
    -DCMAKE_BUILD_TYPE=Release \
    -DARM_TOOLCHAIN_OPT="" \
    -DAIR_RUNTIME_TARGETS="x86_64" \
    -DBUILD_SHARED_LIBS=OFF \
    -DENABLE_RUN_XRT_TESTS=ON \
    -DLLVM_USE_LINKER=lld \
    |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
