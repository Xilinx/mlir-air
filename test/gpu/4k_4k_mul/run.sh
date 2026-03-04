#!/usr/bin/env bash
#===- run.sh ---------------------------------------*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===------------------------------------------------------------------===//
#
# Run the 4k x 4k matrix multiplication example on GPU.
# Assumes environment is set up via: source utils/env_setup_gpu.sh install
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR="${TMPDIR:-/tmp/air_4k_mul}"
mkdir -p "$TMPDIR"

# Step 1: AIR to ROCDL
air-opt "$SCRIPT_DIR/air_sync.mlir" -air-to-rocdl -o "$TMPDIR/mul_gpu.mlir"

# Step 2: GPU kernel outlining
air-opt "$TMPDIR/mul_gpu.mlir" -air-gpu-outlining -o "$TMPDIR/mul_gpu_outline.mlir"

# Step 3: LLVM lowering
mlir-opt "--pass-pipeline=builtin.module(func.func(lower-affine, convert-linalg-to-loops, convert-scf-to-cf), gpu-kernel-outlining)" \
    "$TMPDIR/mul_gpu_outline.mlir" -o "$TMPDIR/mul_gpu_outline_llvm.mlir"

# Step 4: ROCDL binary generation
mlir-opt "--pass-pipeline=builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary, func.func(gpu-async-region),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" \
    "$TMPDIR/mul_gpu_outline_llvm.mlir" -o "$TMPDIR/mul_gpu_final.mlir"

# Step 5: Run
LLVM_LIB_DIR="${LLVM_INSTALL_DIR:-$(dirname "$(which mlir-opt)")/../lib}"
AIRGPU_LIB="${MLIR_AIR_INSTALL_DIR:-$(dirname "$(which air-opt)")/../lib}/libairgpu.so"

mlir-runner --entry-point-result=void \
    --shared-libs="$LLVM_LIB_DIR/libmlir_rocm_runtime.so" \
    --shared-libs="$AIRGPU_LIB" \
    "$TMPDIR/mul_gpu_final.mlir"
