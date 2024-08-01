#!/usr/bin/env bash
##===- utils/setup_python_packages.sh - Setup python packages for mlir-air build --*- Script -*-===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script sets up and installs the required python packages to build mlir-air.
#
# source ./setup_python_packages.sh
#
##===----------------------------------------------------------------------===##

python3 -m venv sandbox
source sandbox/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r utils/requirements.txt

python3 -m pip -q download llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
unzip -q llvm_aie*.whl
rm -rf llvm_aie*.whl
rm -rf llvm_aie-*
rm -rf llvm_aie.libs
python3 -m pip install https://github.com/makslevental/mlir-python-extras/archive/d84f05582adb2eed07145dabce1e03e13d0e29a6.zip