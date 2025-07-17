#!/usr/bin/env bash
#set -x

##===- utils/build-mlir-air-using-wheels.sh - Build mlir-air --*- Script -*-===##
# 
# Copyright (C) 2025, Advanced Micro Devices, Inc.
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
# <xrt dir>    - required, xrt dir.
# <build dir>    - optional, build dir name, default is 'build'
# <install dir>  - optional, install dir name, default is 'install'
#
##===----------------------------------------------------------------------===##

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"

# Install llvm from wheel
VERSION=$($(dirname ${SCRIPT_PATH})/clone-llvm.sh --get-wheel-version)

mkdir -p my_install
pushd my_install
pip -q download mlir==$VERSION \
    -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
unzip -q -u mlir-*.whl
popd
WHL_MLIR_DIR=`realpath my_install/mlir`
echo "WHL_MLIR DIR: $WHL_MLIR_DIR"

# Install mlir-aie dependence: mlir-python-extras
MLIR_PYTHON_EXTRAS_COMMIT_HASH=$($(dirname ${SCRIPT_PATH})/clone-mlir-aie.sh --get-mlir-python-extras-version)
HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install git+https://github.com/makslevental/mlir-python-extras@$MLIR_PYTHON_EXTRAS_COMMIT_HASH -f https://github.com/llvm/eudsl/releases/expanded_assets/latest

# Install mlir-aie from wheel
pushd my_install
MLIR_AIE_VERSION=$($(dirname ${SCRIPT_PATH})/clone-mlir-aie.sh --get-wheel-version)
python3 -m pip install mlir_aie==$MLIR_AIE_VERSION -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2/
popd
export MLIR_AIE_INSTALL_DIR="$(python3 -m pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie"
echo "WHL_AIE DIR: $MLIR_AIE_INSTALL_DIR"

# Environment variables
export PATH=${MLIR_AIE_INSTALL_DIR}/bin:${PATH} 
export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

# Install llvm-aie
python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
PEANO_INSTALL_DIR="$(pip show llvm-aie | grep ^Location: | awk '{print $2}')/llvm-aie"
echo "WHL_LLVM_AIE DIR: $PEANO_INSTALL_DIR"

# Install modulesXilinx
pushd my_install
git clone https://github.com/Xilinx/cmakeModules.git
CMAKEMODULES_DIR=$(pwd)/cmakeModules
popd

# Build mlir-air
XRT_DIR=`realpath $1`
BUILD_DIR=${2:-"build"}
INSTALL_DIR=${3:-"install"}

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
    -DLLVM_DIR=${WHL_MLIR_DIR}/lib/cmake/llvm \
    -DMLIR_DIR=${WHL_MLIR_DIR}/lib/cmake/mlir \
    -DLLVM_EXTERNAL_LIT=$(which lit) \
    -DCMAKE_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86_64.cmake \
    -Dx86_64_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86_64.cmake \
    -DAIE_DIR=${MLIR_AIE_INSTALL_DIR}/lib/cmake/aie \
    -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11 \
    -DPython_FIND_VIRTUALENV=ONLY \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DLibXAIE_ROOT=${MLIR_AIE_INSTALL_DIR}/runtime_lib/x86_64/xaiengine/ \
    -DXRT_LIB_DIR=${XRT_DIR}/lib \
    -DXRT_BIN_DIR=${XRT_DIR}/bin \
    -DXRT_INCLUDE_DIR=${XRT_DIR}/include \
    -DCMAKE_BUILD_TYPE=Release \
    -DAIR_RUNTIME_TARGETS="x86_64" \
    -DBUILD_SHARED_LIBS=OFF \
    -DENABLE_RUN_XRT_TESTS=ON \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_ENABLE_ASSERTIONS=on \
    -DPEANO_INSTALL_DIR=${PEANO_INSTALL_DIR} \
    |& tee cmake.log

ninja |& tee ninja.log
ninja install |& tee ninja-install.log
