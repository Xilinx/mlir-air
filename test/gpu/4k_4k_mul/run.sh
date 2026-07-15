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

# Auto-detect the GPU chip target. Override via `GFX_TARGET=gfx<NNN> ./run.sh`
# for cross-compilation or on heterogeneous systems.
# Detection order:
#   1. amdgpu-arch (ROCm's clang tool; ships at /opt/rocm/llvm/bin/).
#   2. KFD topology: gfx_target_version is major*10000 + minor*100 + step,
#      and the gfx name is gfx<major><minor><step> (e.g. 90500 -> gfx950).
if [ -z "${GFX_TARGET:-}" ]; then
  AMDGPU_ARCH_BIN=$(command -v amdgpu-arch || echo /opt/rocm/llvm/bin/amdgpu-arch)
  GFX_TARGET=$("$AMDGPU_ARCH_BIN" 2>/dev/null | head -1 || true)
  if [ -z "$GFX_TARGET" ]; then
    GFX_TARGET=$(awk '/^gfx_target_version [1-9]/{v=$2; printf "gfx%d%d%d", v/10000, (v%10000)/100, v%100; exit}' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null || true)
  fi
fi
if [ -z "$GFX_TARGET" ]; then
  echo "ERROR: Could not auto-detect GFX_TARGET. Set it explicitly, e.g. GFX_TARGET=gfx950 $0" >&2
  exit 1
fi
echo "GFX_TARGET=$GFX_TARGET"

# Step 1: AIR to ROCDL
air-opt "$SCRIPT_DIR/air_sync.mlir" -air-to-rocdl -o "$TMPDIR/mul_gpu.mlir"

# Step 2: GPU kernel outlining
air-opt "$TMPDIR/mul_gpu.mlir" -air-gpu-outlining -o "$TMPDIR/mul_gpu_outline.mlir"

# Step 3: LLVM lowering
mlir-opt "--pass-pipeline=builtin.module(func.func(lower-affine, convert-linalg-to-loops, convert-scf-to-cf), gpu-kernel-outlining)" \
    "$TMPDIR/mul_gpu_outline.mlir" -o "$TMPDIR/mul_gpu_outline_llvm.mlir"

# Step 4: ROCDL binary generation
mlir-opt "--pass-pipeline=builtin.module(rocdl-attach-target{chip=$GFX_TARGET O=3},gpu.module(convert-gpu-to-rocdl{chipset=$GFX_TARGET runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary, func.func(gpu-async-region),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" \
    "$TMPDIR/mul_gpu_outline_llvm.mlir" -o "$TMPDIR/mul_gpu_final.mlir"

# Step 5: Run
LLVM_LIB_DIR="${LLVM_INSTALL_DIR:+$LLVM_INSTALL_DIR/lib}"
LLVM_LIB_DIR="${LLVM_LIB_DIR:-$(dirname "$(which mlir-opt)")/../lib}"
MLIR_AIR_LIB_DIR="${MLIR_AIR_INSTALL_DIR:+$MLIR_AIR_INSTALL_DIR/lib}"
MLIR_AIR_LIB_DIR="${MLIR_AIR_LIB_DIR:-$(dirname "$(which air-opt)")/../lib}"
AIRGPU_LIB="$MLIR_AIR_LIB_DIR/libairgpu.so"

mlir-runner --entry-point-result=void \
    --shared-libs="$LLVM_LIB_DIR/libmlir_rocm_runtime.so" \
    --shared-libs="$AIRGPU_LIB" \
    "$TMPDIR/mul_gpu_final.mlir"
