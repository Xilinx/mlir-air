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

export PIP_FIND_LINKS="https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels"

MLIR_DISTRO="18.0.0.2024011316+850f713e"
MLIR_AIE_DISTRO="0.0.1.2024011721+d960ffd5"

cd $MLIR_WHL_DIR
pip download mlir_no_rtti==${MLIR_DISTRO}
find . -type f -name "mlir_no_rtti-${MLIR_DISTRO}*" -print0 | xargs -0 unzip -n
cd ../$MLIR_AIE_WHL_DIR
pip download mlir_aie_no_rtti==${MLIR_AIE_DISTRO}
find . -type f -name "mlir_aie_no_rtti-${MLIR_AIE_DISTRO}*" -print0 | xargs -0 unzip -n
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
    2>&1 | tee cmake.log

ninja 2>&1 | tee ninja.log
ninja install 2>&1 | tee ninja-install.log
