#!/usr/bin/env bash

##===- utils/clone-llvm.sh - Clone LLVM ---------------------*- Script -*-===##
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script checks out LLVM.  We use this instead of a git submodule to avoid
# excessive copies of the LLVM tree.
#
# This script is called from the github workflows.
#
##===----------------------------------------------------------------------===##

export commithash=e31ee6417c33a6e2f0e8440b1a86d5365279ad68

git clone --depth 1 https://github.com/llvm/llvm-project.git llvm
pushd llvm
git fetch --depth=1 origin $commithash
git checkout $commithash
popd

