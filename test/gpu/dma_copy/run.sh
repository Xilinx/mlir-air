#!/usr/bin/env bash
#===- run.sh - DMA copy e2e test --------------------------*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===------------------------------------------------------------------===//
#
# Compile and run the DMA data movement e2e test on GPU.
# Assumes environment is set up via: source utils/env_setup_gpu.sh install
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR="${TMPDIR:-/tmp/air_dma_copy}"
mkdir -p "$TMPDIR"

echo "Step 1: AIR to ROCDL"
air-opt "$SCRIPT_DIR/air_dma_copy.mlir" -air-to-rocdl -o "$TMPDIR/step1.mlir"

echo "Step 2: GPU kernel outlining"
air-opt "$TMPDIR/step1.mlir" -air-gpu-outlining -o "$TMPDIR/step2.mlir"

echo "Step 3: LLVM lowering"
mlir-opt "--pass-pipeline=builtin.module(func.func(lower-affine, convert-linalg-to-loops, convert-scf-to-cf), gpu-kernel-outlining)" \
    "$TMPDIR/step2.mlir" -o "$TMPDIR/step3.mlir"

echo "Step 4: ROCDL binary generation"
mlir-opt "--pass-pipeline=builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary, func.func(gpu-async-region),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" \
    "$TMPDIR/step3.mlir" -o "$TMPDIR/step4.mlir"

echo "Step 5: Running on GPU"
LLVM_LIB_DIR="${LLVM_INSTALL_DIR:-$(dirname "$(which mlir-opt)")/..}/lib"
AIRGPU_LIB="${MLIR_AIR_INSTALL_DIR:-$(dirname "$(which air-opt)")/..}/lib/libairgpu.so"

mlir-runner --entry-point-result=void \
    --shared-libs="$LLVM_LIB_DIR/libmlir_rocm_runtime.so" \
    --shared-libs="$AIRGPU_LIB" \
    --shared-libs="$LLVM_LIB_DIR/libmlir_runner_utils.so" \
    --shared-libs="$LLVM_LIB_DIR/libmlir_c_runner_utils.so" \
    "$TMPDIR/step4.mlir"
