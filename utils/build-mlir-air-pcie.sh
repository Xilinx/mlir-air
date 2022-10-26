#!/usr/bin/env bash
#set -x

##===- utils/build-mlir-air-pcie.sh - Build mlir-air --*- Script -*-===##
# 
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# 
##===----------------------------------------------------------------------===##
#
# This script builds mlir-air given the <llvm dir>, <cmakeModules dir>,
# and <mlir-aie dir>. Assuming they are all in the same subfolder,
# it would look like:
#
# build-mlir-air-pcie.sh <llvm dir> <cmakeModules dir> <mlir-aie dir>
#     <mlir-air dir> <build dir> <install dir>
#
# e.g. build-mlir-air-pcie.sh /scratch/llvm /scratch/mlir-aie
#          /scratch/cmakeModules/cmakeModulesXilinx
#
# <cmakeModules dir> - cmakeModules, absolute directory
# <mlir-aie dir>     - mlir-aie, absolute directory
#
# <mlir-air dir> - optional, mlir-air repo name, default is 'mlir-air'
# <build dir>    - optional, mlir-air/build dir name, default is 'build-pcie'
# <install dir>  - optional, mlir-air/install dir name, default is 'install-pcie'
#
##===----------------------------------------------------------------------===##

if [ "$#" -lt 3 ]; then
    echo "ERROR: Needs at least 3 arguments for <llvm dir>, <cmakeModules dir> and <mlir-aie dir>."
    exit 1
fi
LLVM_DIR=$1
CMAKEMODULES_DIR=$2
MLIR_AIE_DIR=$3

#LLVM_DIR=${2:-"./llvm"}
#CMAKEMODULES_DIR=${3:-"./cmakeModules/cmakeModulesXilinx"}

MLIR_AIR_DIR=${4:-"mlir-air"}
BUILD_DIR=${5:-"build-pcie"}
INSTALL_DIR=${6:-"install-pcie"}

mkdir -p $MLIR_AIR_DIR/$BUILD_DIR
mkdir -p $MLIR_AIR_DIR/$INSTALL_DIR
cd $MLIR_AIR_DIR/$BUILD_DIR

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $1}'`

cmake .. \
	-GNinja \
	-DCMAKE_INSTALL_PREFIX=../install-pcie \
	-DCMAKE_TOOLCHAIN_FILE=../cmake/modules/toolchain_x86.cmake \
	-DLLVM_DIR=${LLVM_DIR}/build/lib/cmake/llvm \
	-DMLIR_DIR=${LLVM_DIR}/build/lib/cmake/mlir \
	-DAIE_DIR=${MLIR_AIE_DIR}/build/lib/cmake/aie \
	-Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 \
	-DVitisSysroot="" \
	-DARM_TOOLCHAIN_OPT="" \
  -DBUILD_AIR_PCIE=ON \
	-DBUILD_SHARED_LIBS=OFF \
	-DLLVM_USE_LINKER=lld \
	-DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/ \
	|& tee cmake.log


ninja |& tee ninja.log
ninja install |& tee ninja-install.log
