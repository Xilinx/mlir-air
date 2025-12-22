#!/usr/bin/env bash
##===- utils/setup_python_packages.sh - Setup python packages for mlir-air build --*- Script -*-===##
# 
# SPDX-License-Identifier: MIT
# 
##===----------------------------------------------------------------------===##
#
# This script sets up and installs the required python packages to build mlir-air.
#
# source ./setup_python_packages.sh
#
##===----------------------------------------------------------------------===##

# Set up python venv 'sandbox'
python3 -m venv sandbox
source sandbox/bin/activate
# Install essential python packages
python3 -m pip install --upgrade pip
python3 -m pip install -r utils/requirements.txt
# Install python packages needed by MLIR-AIE's python bindings
EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=aie python3 -m pip install -r utils/requirements_extras.txt
