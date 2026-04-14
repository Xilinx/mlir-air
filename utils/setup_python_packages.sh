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
# Install runtime dependencies and build/dev/test dependencies.
# End users installing the mlir-air wheel only get runtime deps;
# this script installs both since it sets up a build environment.
python3 -m pip install --upgrade pip
python3 -m pip install -r utils/requirements.txt
python3 -m pip install -r utils/requirements_dev.txt
