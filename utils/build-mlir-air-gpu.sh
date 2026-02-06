#!/usr/bin/env bash

##===- utils/build-mlir-air-gpu.sh - Build mlir-air for GPU --*- Script -*-===##
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
##===----------------------------------------------------------------------===##
#
# This script builds mlir-air for GPU backend (ROCDL) without AIE dependency.
#
# Prerequisites:
#   - LLVM/MLIR must be built with full library exports (no LLVM_DISTRIBUTION_COMPONENTS)
#   - Use utils/build-llvm-local.sh (updated version) to build LLVM
#
# Usage:
#   build-mlir-air-gpu.sh <llvm install dir> [build dir] [install dir]
#
# Arguments:
#   <llvm install dir> - Path to LLVM installation (with lib/cmake/llvm and lib/cmake/mlir)
#   <build dir>        - Optional, build directory name, default is 'build'
#   <install dir>      - Optional, install directory name, default is 'install'
#
# Example:
#   ./utils/build-mlir-air-gpu.sh llvm/install build install
#
##===----------------------------------------------------------------------===##

set -e

if [ "$#" -lt 1 ]; then
    echo "ERROR: Needs at least 1 argument for <llvm install dir>."
    echo "Usage: $0 <llvm install dir> [build dir] [install dir]"
    exit 1
fi

LLVM_DIR="$1"
BUILD_DIR=${2:-"build"}
INSTALL_DIR=${3:-"install"}

# Auto-detect LLVM install directory
# Check if user passed source dir (llvm) vs install dir (llvm/install)
if [ -f "${LLVM_DIR}/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    LLVM_INSTALL_DIR=$(realpath "$LLVM_DIR")
elif [ -f "${LLVM_DIR}/install/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    LLVM_INSTALL_DIR=$(realpath "${LLVM_DIR}/install")
    echo "Note: Using ${LLVM_INSTALL_DIR} (auto-detected install subdirectory)"
else
    echo "ERROR: Could not find MLIRConfig.cmake"
    echo "Checked: ${LLVM_DIR}/lib/cmake/mlir/"
    echo "Checked: ${LLVM_DIR}/install/lib/cmake/mlir/"
    echo "Please provide the path to the LLVM installation directory."
    exit 1
fi

echo "=== Building MLIR-AIR for GPU (without AIE) ==="
echo "LLVM_INSTALL_DIR: ${LLVM_INSTALL_DIR}"
echo "BUILD_DIR: ${BUILD_DIR}"
echo "INSTALL_DIR: ${INSTALL_DIR}"

mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR"
cd "$BUILD_DIR"

# Find pybind11
PYTHON_ROOT=$(pip3 show pybind11 2>/dev/null | grep Location | awk '{print $2}' || echo "")
if [ -z "$PYTHON_ROOT" ]; then
    echo "WARNING: pybind11 not found. Python bindings will be disabled."
    PYBIND11_ARG=""
    PYTHON_BINDINGS_ARG="-DAIE_ENABLE_BINDINGS_PYTHON=OFF"
else
    PYBIND11_ARG="-Dpybind11_DIR=${PYTHON_ROOT}/pybind11/share/cmake/pybind11"
    # Check if LLVM was built with Python bindings
    if [ -f "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRPythonSources.cmake" ] || [ -d "${LLVM_INSTALL_DIR}/python" ]; then
        PYTHON_BINDINGS_ARG="-DAIE_ENABLE_BINDINGS_PYTHON=ON"
    else
        echo "WARNING: LLVM Python bindings not found. Python bindings will be disabled."
        PYTHON_BINDINGS_ARG="-DAIE_ENABLE_BINDINGS_PYTHON=OFF"
    fi
fi

# Find lit
LIT_PATH=$(which lit 2>/dev/null || echo "")
if [ -z "$LIT_PATH" ]; then
    echo "WARNING: lit not found. Tests may not work."
    LIT_ARG=""
else
    LIT_ARG="-DLLVM_EXTERNAL_LIT=${LIT_PATH}"
fi

CMAKE_ARGS="-GNinja"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release"
CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=../${INSTALL_DIR}"
CMAKE_ARGS="$CMAKE_ARGS -DLLVM_DIR=${LLVM_INSTALL_DIR}/lib/cmake/llvm"
CMAKE_ARGS="$CMAKE_ARGS -DMLIR_DIR=${LLVM_INSTALL_DIR}/lib/cmake/mlir"

# Disable AIE - this is the key for GPU-only build
CMAKE_ARGS="$CMAKE_ARGS -DAIR_ENABLE_AIE=OFF"
# Enable GPU/ROCDL backend
CMAKE_ARGS="$CMAKE_ARGS -DAIR_ENABLE_GPU=ON"

# Python settings
CMAKE_ARGS="$CMAKE_ARGS -DPython3_FIND_VIRTUALENV=ONLY"
CMAKE_ARGS="$CMAKE_ARGS -DPython_FIND_VIRTUALENV=ONLY"
CMAKE_ARGS="$CMAKE_ARGS ${PYBIND11_ARG}"
CMAKE_ARGS="$CMAKE_ARGS ${PYTHON_BINDINGS_ARG}"
CMAKE_ARGS="$CMAKE_ARGS ${LIT_ARG}"

# Build settings
CMAKE_ARGS="$CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF"
CMAKE_ARGS="$CMAKE_ARGS -DLLVM_ENABLE_ASSERTIONS=ON"

# Use lld if available
if command -v lld &> /dev/null; then
    CMAKE_ARGS="$CMAKE_ARGS -DLLVM_USE_LINKER=lld"
fi

# Use ccache if available
if command -v ccache &> /dev/null; then
    CMAKE_ARGS="$CMAKE_ARGS -DLLVM_CCACHE_BUILD=ON"
fi

# Detect compiler
if command -v clang &> /dev/null; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER=clang"
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_COMPILER=clang++"
fi

echo ""
echo "=== Running CMake ==="
echo "cmake $CMAKE_ARGS .."
echo ""

cmake $CMAKE_ARGS .. 2>&1 | tee cmake.log

if [ ! -f "build.ninja" ]; then
    echo ""
    echo "ERROR: CMake configuration failed. Check cmake.log for details."
    exit 1
fi

echo ""
echo "=== Building with Ninja ==="
ninja 2>&1 | tee ninja.log

echo ""
echo "=== Installing ==="
ninja install 2>&1 | tee ninja-install.log

echo ""
echo "=== Build Complete ==="
echo "Install location: $(realpath ../${INSTALL_DIR})"
