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

commithash=47ff7d38a89372c02e22515e61a696f2a3e93013

# Avoid checking out to spare time since we switch to another branch later
git clone --depth 1 --no-checkout https://github.com/Xilinx/mlir-aie.git mlir-aie
(
  cd mlir-aie
  git fetch --depth=1 origin $commithash
  git checkout $commithash
  git submodule update --depth 1 --recursive --init
)
