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

export commithash=4b06b01e62ab0cfd059f47dceeb6cd0fb96304c9
target_dir=llvm

# clone llvm if it is not there already
if [[ ! -d $target_dir ]]; then
  git clone --depth 1 https://github.com/llvm/llvm-project.git $target_dir
fi

pushd $target_dir
git fetch --depth=1 origin $commithash
git checkout $commithash
# Make mlir_async_runtime library's symbol visible
# so that we can link to this library in channel sim tests
sed -i '/set_property(TARGET mlir_async_runtime PROPERTY CXX_VISIBILITY_PRESET hidden)/d' ./mlir/lib/ExecutionEngine/CMakeLists.txt
popd
