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

export HASH=bbd8ce98c5db2e9d3bbb51eacd71e67a0fbeb68a

git clone --depth 1 https://github.com/Xilinx/cmakeModules cmakeModules/cmakeModulesXilinx
export CMAKE_MODULE_PATH=`pwd`/cmakeModules/cmakeModulesXilinx

git clone --depth 1 https://github.com/Xilinx/mlir-aie.git mlir-aie
pushd mlir-aie
git fetch --depth=1 origin $HASH
git checkout $HASH
popd
