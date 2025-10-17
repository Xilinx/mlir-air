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
# <peano dir>        - llvm-aie
#
# <XRT dir>          - XRT build runtime
#
# <build dir>    - optional, build dir name, default is 'build'
# <install dir>  - optional, install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 3 ]; then
    echo "ERROR: Needs at least 3 arguments for <llvm dir>, <cmakeModules dir> and <mlir-aie dir>."
    exit 1
fi
LLVM_DIR=`realpath $1`
CMAKEMODULES_DIR=`realpath $2`
MLIR_AIE_DIR=`realpath $3`

PEANO_INSTALL_DIR=`realpath $4`

XRT_DIR=`realpath $5`

BUILD_DIR=${6:-"build"}
INSTALL_DIR=${7:-"install"}

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $2}'`

CMAKE_ARGS="-GNinja"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER=clang"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER=clang++"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=../${INSTALL_DIR}"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/"
CMAKE_ARGS="$CMAKE_ARGS -DLLVM_EXTERNAL_LIT=$(which lit)"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86_64.cmake"
CMAKE_ARGS="$CMAKE_ARGS -Dx86_64_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86_64.cmake"
CMAKE_ARGS="$CMAKE_ARGS -DLLVM_DIR=${LLVM_DIR}/lib/cmake/llvm"
CMAKE_ARGS="$CMAKE_ARGS -DMLIR_DIR=${LLVM_DIR}/lib/cmake/mlir"
CMAKE_ARGS="$CMAKE_ARGS -DAIE_DIR=${MLIR_AIE_DIR}/lib/cmake/aie"
CMAKE_ARGS="$CMAKE_ARGS -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11"
CMAKE_ARGS="$CMAKE_ARGS -DPython_FIND_VIRTUALENV=ONLY"
CMAKE_ARGS="$CMAKE_ARGS -DPython3_FIND_VIRTUALENV=ONLY"
CMAKE_ARGS="$CMAKE_ARGS -DXRT_LIB_DIR=${XRT_DIR}/lib"
CMAKE_ARGS="$CMAKE_ARGS -DXRT_BIN_DIR=${XRT_DIR}/bin"
CMAKE_ARGS="$CMAKE_ARGS -DXRT_INCLUDE_DIR=${XRT_DIR}/include"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release"
CMAKE_ARGS="$CMAKE_ARGS -DAIR_RUNTIME_TARGETS=x86_64"
CMAKE_ARGS="$CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF"
CMAKE_ARGS="$CMAKE_ARGS -DENABLE_RUN_XRT_TESTS=ON"
CMAKE_ARGS="$CMAKE_ARGS -DLLVM_ENABLE_ASSERTIONS=on"
CMAKE_ARGS="$CMAKE_ARGS -DPEANO_INSTALL_DIR=${PEANO_INSTALL_DIR}"

if [ -x "$(command -v lld)" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DLLVM_USE_LINKER=lld"
fi

if [ -x "$(command -v ccache)" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DLLVM_CCACHE_BUILD=ON"
fi

cmake $CMAKE_ARGS .. |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
