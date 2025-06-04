##===- utils/env_setup.sh - Setup mlir-aie env post build to compile mlir-aie designs --*- Script -*-===##
# 
# SPDX-License-Identifier: MIT
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

export MLIR_AIR_INSTALL_DIR=`realpath $1`
export MLIR_AIE_INSTALL_DIR=`realpath $2`
export LLVM_INSTALL_DIR=`realpath $3`

export PATH=${MLIR_AIR_INSTALL_DIR}/bin:${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_INSTALL_DIR}/bin:${PATH} 
export PYTHONPATH=${MLIR_AIR_INSTALL_DIR}/python:${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH} 
export LD_LIBRARY_PATH=${MLIR_AIR_INSTALL_DIR}/lib:${MLIR_AIE_INSTALL_DIR}/lib:${LLVM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
