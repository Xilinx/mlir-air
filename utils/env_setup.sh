#!/bin/bash
##===- utils/env_setup.sh - Setup mlir-aie env post build to compile mlir-aie designs --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script sets up the environment to run the mlir-aie build tools.
#
# source env_setup.sh <mlir-aie install dir> <llvm install dir>
#
# e.g. source env_setup.sh /scratch/mlir-aie/install /scratch/llvm/install
#
##===----------------------------------------------------------------------===##

if [ "$#" -ne 3 ]; then
    echo "ERROR: Needs 3 arguments for <mlir-air install dir> <mlir-aie install dir> and <llvm install dir>"
    exit 1
fi

export MLIR_AIR_INSTALL_DIR=$1
export MLIR_AIE_INSTALL_DIR=$2
export LLVM_INSTALL_DIR=$3

export PATH=${MLIR_AIR_INSTALL_DIR}/bin:${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_INSTALL_DIR}/bin:${PATH} 
export PYTHONPATH=${MLIR_AIR_INSTALL_DIR}/python:${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH} 
export LD_LIBRARY_PATH=${MLIR_AIR_INSTALL_DIR}/lib:${MLIR_AIE_INSTALL_DIR}/lib:${LLVM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

