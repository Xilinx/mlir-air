#!/usr/bin/env bash

##===- utils/github-build-mlir-aie.sh -----------------*- Script -*-===##
# 
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script builds a specific version of mlir-aie.
#
# This script is intended to be called from the github workflows.
#
##===----------------------------------------------------------------------===##

MLIR_AIE_DIR="mlir-aie"
BUILD_DIR="build"
INSTALL_DIR="install"

mkdir -p $MLIR_AIE_DIR/$BUILD_DIR
mkdir -p $MLIR_AIE_DIR/$INSTALL_DIR
pushd $MLIR_AIE_DIR/$BUILD_DIR

cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAIE_COMPILER=NONE \
    -DAIE_LINKER=NONE \
    -DHOST_COMPILER=NONE \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_MODULE_PATH=`pwd`/../cmake/modulesXilinx \
    -DMLIR_DIR=`pwd`/../../llvm/install/lib/cmake/mlir/ \
    -DLLVM_DIR=`pwd`/../../llvm/install/lib/cmake/llvm/ \
    -DCMAKE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang-12 \
    -DCMAKE_CXX_COMPILER=clang++-12 \
    -DLLVM_EXTERNAL_LIT=`pwd`/../../llvm/build/bin/llvm-lit \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLibXAIE_x86_64_DIR=`pwd`/../../aienginev2/install/lib \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../$INSTALL_DIR

ec=$?
if [ $ec -ne 0 ]; then
    echo "CMake Configuration Error"
    exit $ec
fi

ninja install

ec=$?
if [ $ec -ne 0 ]; then
    echo "Ninja Build Error"
    exit $ec
fi

popd
