#!/usr/bin/env bash

##===- utils/build-mlir-aie.sh - Build mlir-aie --*- Script -*-===##
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

MLIR_AIE_DIR="mlir-aie"
BUILD_DIR="build"
INSTALL_DIR="install"

HASH=8178816f2f13beccc072a2e0b1001abab548fd84

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
