#!/usr/bin/env bash

##===- utils/clone-mlir-aie.sh - Clone MLIR-AIE --------------*- Script -*-===##
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script checks out MLIR-AIE.  We use this instead of a git submodule to
# manage commithash synchronization with LLVM.
#
# This script is called from the github workflows.
#
##===----------------------------------------------------------------------===##

commithash=47ff7d38a89372c02e22515e61a696f2a3e93013
project_dir=mlir-aie

here=$PWD

# Use --mlir-aie-worktree <path-of-local-mlir-aie-repository> to reuse
# some existing local mlir-aie git repository
if [ x"$1" == x--mlir-aie-worktree ]; then
  git_central_mlir_aie_repo_dir="$2"
  (
    cd $git_central_mlir_aie_repo_dir
    # Remove an existing worktree here just in case we are iterating
    # on the script
    git worktree remove --force "$here"/$project_dir
    # Use force just in case there are various experimental iterations
    # after you have removed the llvm directory
    git worktree add --force "$here"/$project_dir $commithash
  )
else
  # Avoid checking out to spare time since we switch to another branch later
  git clone --depth 1 --no-checkout https://github.com/Xilinx/$project_dir.git
  (
    cd $project_dir
    git fetch --depth=1 origin $commithash
    git checkout $commithash
    git submodule update --depth 1 --recursive --init
  )
fi
(
  cd $project_dir
  git submodule update --depth 1 --recursive --init
)
