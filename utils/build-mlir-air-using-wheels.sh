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
# Usage: build-mlir-air-using-wheels.sh [--xrt-dir <xrt_path>] [<build_dir>] [<install_dir>]
#
# --xrt-dir <path> - optional, path to XRT installation directory
# <build dir>      - optional, build dir name, default is 'build'
# <install dir>    - optional, install dir name, default is 'install'
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

# Install mlir-aie from wheel
pushd my_install
MLIR_AIE_VERSION=$($(dirname ${SCRIPT_PATH})/clone-mlir-aie.sh --get-wheel-version)
python3 -m pip install mlir_aie==$MLIR_AIE_VERSION -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-3/
popd
export MLIR_AIE_INSTALL_DIR="$(python3 -m pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie"
echo "WHL_AIE DIR: $MLIR_AIE_INSTALL_DIR"

# Environment variables
export PATH=${MLIR_AIE_INSTALL_DIR}/bin:${PATH} 
export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

# Install llvm-aie
# TODO: Use nightly latest llvm-aie once it is fixed
python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025071101+b3cd09d3-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
PEANO_INSTALL_DIR="$(pip show llvm-aie | grep ^Location: | awk '{print $2}')/llvm-aie"
echo "WHL_LLVM_AIE DIR: $PEANO_INSTALL_DIR"

# Install modulesXilinx
pushd my_install
git clone https://github.com/Xilinx/cmakeModules.git
CMAKEMODULES_DIR=$(pwd)/cmakeModules
popd

# Parse command-line arguments
XRT_DIR=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --xrt-dir)
      if [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "Error: --xrt-dir requires a path argument"
        exit 1
      fi
      XRT_DIR=$(realpath "$2")
      if [ ! -d "$XRT_DIR" ]; then
        echo "Error: XRT directory does not exist: $XRT_DIR"
        exit 1
      fi
      shift 2
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Build mlir-air
BUILD_DIR=${POSITIONAL_ARGS[0]:-"build"}
INSTALL_DIR=${POSITIONAL_ARGS[1]:-"install"}

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR
cd $BUILD_DIR

PYTHON_ROOT=`pip3 show pybind11 | grep Location | awk '{print $2}'`

CMAKE_ARGS="-GNinja"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER=clang"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER=clang++"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=../${INSTALL_DIR}"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_MODULE_PATH=${CMAKEMODULES_DIR}/"
CMAKE_ARGS="$CMAKE_ARGS -DLLVM_DIR=${WHL_MLIR_DIR}/lib/cmake/llvm"
CMAKE_ARGS="$CMAKE_ARGS -DMLIR_DIR=${WHL_MLIR_DIR}/lib/cmake/mlir"
CMAKE_ARGS="$CMAKE_ARGS -DLLVM_EXTERNAL_LIT=$(which lit)"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86_64.cmake"
CMAKE_ARGS="$CMAKE_ARGS -Dx86_64_TOOLCHAIN_FILE=`pwd`/../cmake/modules/toolchain_x86_64.cmake"
CMAKE_ARGS="$CMAKE_ARGS -DAIE_DIR=${MLIR_AIE_INSTALL_DIR}/lib/cmake/aie"
CMAKE_ARGS="$CMAKE_ARGS -Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11"
CMAKE_ARGS="$CMAKE_ARGS -DPython_FIND_VIRTUALENV=ONLY"
CMAKE_ARGS="$CMAKE_ARGS -DPython3_FIND_VIRTUALENV=ONLY"

# Add XRT-related arguments only if XRT directory is provided
if [ -n "$XRT_DIR" ]; then
  echo "Building with XRT support from: $XRT_DIR"
  CMAKE_ARGS="$CMAKE_ARGS -DXRT_LIB_DIR=${XRT_DIR}/lib"
  CMAKE_ARGS="$CMAKE_ARGS -DXRT_BIN_DIR=${XRT_DIR}/bin"
  CMAKE_ARGS="$CMAKE_ARGS -DXRT_INCLUDE_DIR=${XRT_DIR}/include"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_RUN_XRT_TESTS=ON"
else
  echo "Building without XRT support"
fi

CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release"
CMAKE_ARGS="$CMAKE_ARGS -DAIR_RUNTIME_TARGETS=x86_64"
CMAKE_ARGS="$CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF"
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
