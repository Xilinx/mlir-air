#!/usr/bin/env bash
#set -x

##===- utils/build-mlir-air-using-wheels.sh - Build mlir-air --*- Script -*-===##
# 
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# 
##===----------------------------------------------------------------------===##
#
# This script build mlir-air using wheels released from Xilinx/mlir-aie repository. 
#
# Note: released mlir-aie wheels require the system to have gcc version >= 11
#       and Python 3.10 and above. For lit tests, please set -DLLVM_EXTERNAL_LIT
#       path.
#
# <build dir>    - optional, build dir name, default is 'build'
# <install dir>  - optional, install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

BUILD_DIR=${1:-"build"}
INSTALL_DIR=${2:-"install"}
MLIR_WHL_DIR="mlir_wheel"
MLIR_AIE_WHL_DIR="mlir_aie_wheel"

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR
mkdir -p $MLIR_WHL_DIR
mkdir -p $MLIR_AIE_WHL_DIR

MLIR_DISTRO="mlir_no_rtti-18.0.0.2023121521+d36b483-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
MLIR_AIE_DISTRO="mlir_aie_no_rtti-0.0.1.2023121602+5631ba5c-py3-none-manylinux_2_28_x86_64.whl"

cd $MLIR_WHL_DIR
if [ -f ${MLIR_DISTRO} ]; then
  echo "MLIR wheel exists."
else
  wget https://github.com/Xilinx/mlir-aie/releases/download/mlir-distro/${MLIR_DISTRO}
  unzip ${MLIR_DISTRO}
fi
cd ../$MLIR_AIE_WHL_DIR
if [ -f ${MLIR_AIE_DISTRO} ]; then
  echo "MLIR-AIE wheel exists."
else
  wget https://github.com/Xilinx/mlir-aie/releases/download/latest-wheels/${MLIR_AIE_DISTRO}
  unzip ${MLIR_AIE_DISTRO}
fi
cd ..

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $2}'`

MLIR_WHL_DIR_REAL=`realpath "${MLIR_WHL_DIR}/mlir_no_rtti"`
MLIR_AIE_WHL_DIR_REAL=`realpath "${MLIR_AIE_WHL_DIR}/mlir_aie_no_rtti"`

cmake .. \
    -GNinja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
    -DCMAKE_USE_TOOLCHAIN=FALSE \
    -DCMAKE_USE_TOOLCHAIN_AIRHOST=TRUE \
    -DLLVM_DIR=${MLIR_WHL_DIR_REAL}/lib/cmake/llvm \
    -DMLIR_DIR=${MLIR_WHL_DIR_REAL}/lib/cmake/mlir \
    -DAIE_DIR=${MLIR_AIE_WHL_DIR_REAL}/lib/cmake/aie \
    -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_MODULE_PATH="../cmake/modules" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=X86 \
    -DLLVM_HOST_TRIPLE=x86_64-unknown-linux-gnu \
    -DLLVM_ENABLE_RTTI=OFF \
    |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
