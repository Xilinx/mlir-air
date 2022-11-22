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

export commithash=e864ac694540342d5e59f59c525c5082f2594fb8

git clone --depth 1 https://github.com/llvm/llvm-project.git llvm
pushd llvm
git fetch --depth=1 origin $commithash
git checkout $commithash
popd

