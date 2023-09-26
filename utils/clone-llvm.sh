#!/usr/bin/env bash

##===- utils/clone-llvm.sh - Clone LLVM ---------------------*- Script -*-===##
#
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script checks out LLVM.  We use this instead of a git submodule to avoid
# excessive copies of the LLVM tree.
#
# This script is called from the github workflows.
#
##===----------------------------------------------------------------------===##

# The LLVM commit the project depends on
commithash=11c3b979e6512b00a5bd9c3e0d4ed986cf500630
# It is not clear why we need a branch
branch=air-2022.12

here=$PWD

# Use --llvm-worktree <path-of-local-LLVM-repo> to reuse some existing
# local LLVM git repository
if [ x"$1" == x--llvm-worktree ]; then
  git_central_llvm_repo_dir="$2"
  (
    cd $git_central_llvm_repo_dir
    # Remove an existing worktree here just in case we are iterating
    # on the script
    git worktree remove --force "$here"/llvm
    # Use force just in case there are various experimental iterations
    # after you have removed the llvm directory
    git worktree add --force -B $branch "$here"/llvm $commithash
  )
else
  # Avoid checking out to spare time since we switch to another branch later
  git clone --depth 1 --no-checkout https://github.com/llvm/llvm-project.git llvm
  (
    cd llvm
    # Then fetch the interesting part
    git fetch --depth=1 origin $commithash
    # Create local $branch to current $commithash
    git branch $branch $commithash
    # Switch to the branch
    git switch $branch
  )
fi

# Make mlir_async_runtime library's symbol visible
# so that we can link to this library in channel sim tests
sed -i '/set_property(TARGET mlir_async_runtime PROPERTY CXX_VISIBILITY_PRESET hidden)/d' llvm/mlir/lib/ExecutionEngine/CMakeLists.txt
