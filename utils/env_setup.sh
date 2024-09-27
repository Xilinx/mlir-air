##===- utils/env_setup.sh - Setup mlir-aie env post build to compile mlir-aie designs --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script sets up the environment to run the mlir-aie build tools.
# It must be sourced, not executed.
#
# e.g. source env_setup.sh /scratch/mlir-air/install /scratch/mlir-aie/install /scratch/llvm/install
#
##===----------------------------------------------------------------------===##

if [ "$#" -ne 3 ]; then
	echo "ERROR: Needs 3 arguments for <mlir-air install dir> <mlir-aie install dir> and <llvm install dir>"
	return
fi

# This is populated during the setup_python_packages script
export PEANO_INSTALL_DIR=`realpath llvm-aie`
export MLIR_AIR_INSTALL_DIR=`realpath $1`
export MLIR_AIE_INSTALL_DIR=`realpath $2`
export LLVM_INSTALL_DIR=`realpath $3`

export PATH=`realpath llvm-aie/bin`:${MLIR_AIR_INSTALL_DIR}/bin:${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_INSTALL_DIR}/bin:${PATH}
export PYTHONPATH=${MLIR_AIR_INSTALL_DIR}/python:${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH} 
export LD_LIBRARY_PATH=`realpath llvm-aie/lib`:${MLIR_AIR_INSTALL_DIR}/lib:${MLIR_AIE_INSTALL_DIR}/lib:${LLVM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
