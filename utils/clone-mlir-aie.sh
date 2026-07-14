#!/usr/bin/env bash

##===- utils/clone-mlir-aie.sh - Clone MLIR-AIE --------------*- Script -*-===##
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script checks out MLIR-AIE.  We use this instead of a git submodule to 
# manage commithash synchronization with LLVM.
#
# This script is called from the github workflows.
#
##===----------------------------------------------------------------------===##

export HASH=660e1a0e04097debaf6846b3bba7893a56fb97fd
WHEEL_VERSION=1.3.5.dev52+g${HASH:0:7}

if [ x"$1" == x--get-wheel-version ]; then
  echo $WHEEL_VERSION
  exit 0
fi

MLIR_PYTHON_EXTRAS_SHORTHASH=a736a7d

if [ x"$1" == x--get-mlir-python-extras-version ]; then
  echo $MLIR_PYTHON_EXTRAS_SHORTHASH
  exit 0
fi

target_dir=mlir-aie

if [[ ! -d $target_dir ]]; then
  git clone --depth 1 https://github.com/Xilinx/mlir-aie.git $target_dir
fi

pushd $target_dir
git fetch --depth=1 origin $HASH
git checkout $HASH
git submodule update --init

popd
