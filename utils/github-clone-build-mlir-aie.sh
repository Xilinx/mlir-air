#!/usr/bin/env bash

##===- utils/github-clone-build-mlir-aie.sh -----------------*- Script -*-===##
# 
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script checks out and builds a specific version of mlir-aie.
#
# This script is intended to be called from the github workflows.
#
##===----------------------------------------------------------------------===##

MLIR_AIE_DIR="mlir-aie"
BUILD_DIR="build"
INSTALL_DIR="install"

HASH=ad2cbf5366a2470ae6b543cff53e80bed1179d79

git clone --depth 1 https://github.com/Xilinx/mlir-aie.git $MLIR_AIE_DIR
pushd $MLIR_AIE_DIR
git fetch --depth=1 origin $HASH
git checkout $HASH
popd

mkdir -p $MLIR_AIE_DIR/$BUILD_DIR
mkdir -p $MLIR_AIE_DIR/$INSTALL_DIR
pushd $MLIR_AIE_DIR/$BUILD_DIR

cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DAIE_COMPILER=NONE \
    -DAIE_LINKER=NONE \
    -DHOST_COMPILER=NONE \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_MODULE_PATH=`pwd`/../cmakeModules \
    -DMLIR_DIR=`pwd`/../../llvm/install/lib/cmake/mlir/ \
    -DLLVM_DIR=`pwd`/../../llvm/install/lib/cmake/llvm/ \
    -DCMAKE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_EXTERNAL_LIT=`pwd`/../../llvm/build/bin/llvm-lit \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX=`pwd`/../$INSTALL_DIR

cmake --build . --target install -- -j$(nproc)

popd
