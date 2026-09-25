#!/usr/bin/env bash
#===- run.sh ---------------------------------------*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===------------------------------------------------------------------===//
#
# Run the chiplet-identity reporting test on GPU.
# Assumes environment is set up via: source utils/env_setup_gpu.sh install
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR="${TMPDIR:-/tmp/air_chiplet_identity}"
mkdir -p "$TMPDIR"

# amdgpu-arch reports a full target id (gfx942:sramecc+:xnack-); the chipset
# options below take the bare gfx name, so cut the features off.
if [ -z "${GFX_TARGET:-}" ]; then
  AMDGPU_ARCH_BIN=$(command -v amdgpu-arch || echo /opt/rocm/llvm/bin/amdgpu-arch)
  GFX_TARGET=$("$AMDGPU_ARCH_BIN" 2>/dev/null | head -1 | cut -d: -f1 || true)
fi
if [ -z "$GFX_TARGET" ]; then
  echo "ERROR: could not detect GFX_TARGET; set it explicitly, e.g. GFX_TARGET=gfx942 $0" >&2
  exit 1
fi
echo "GFX_TARGET=$GFX_TARGET"

air-opt "$SCRIPT_DIR/chiplet_identity.mlir" -air-to-rocdl -o "$TMPDIR/s1.mlir"
air-opt "$TMPDIR/s1.mlir" -air-gpu-outlining -o "$TMPDIR/s2.mlir"
mlir-opt "--pass-pipeline=builtin.module(func.func(lower-affine, convert-linalg-to-loops, convert-scf-to-cf), gpu-kernel-outlining)" \
    "$TMPDIR/s2.mlir" -o "$TMPDIR/s3.mlir"
mlir-opt "--pass-pipeline=builtin.module(rocdl-attach-target{chip=$GFX_TARGET O=3},gpu.module(convert-gpu-to-rocdl{chipset=$GFX_TARGET runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary, func.func(gpu-async-region),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" \
    "$TMPDIR/s3.mlir" -o "$TMPDIR/s4.mlir"

LLVM_LIB_DIR="${LLVM_INSTALL_DIR:+$LLVM_INSTALL_DIR/lib}"
LLVM_LIB_DIR="${LLVM_LIB_DIR:-$(dirname "$(which mlir-opt)")/../lib}"
MLIR_AIR_LIB_DIR="${MLIR_AIR_INSTALL_DIR:+$MLIR_AIR_INSTALL_DIR/lib}"
MLIR_AIR_LIB_DIR="${MLIR_AIR_LIB_DIR:-$(dirname "$(which air-opt)")/../lib}"

mlir-runner --entry-point-result=void \
    --shared-libs="$LLVM_LIB_DIR/libmlir_rocm_runtime.so" \
    --shared-libs="$LLVM_LIB_DIR/libmlir_runner_utils.so" \
    --shared-libs="$LLVM_LIB_DIR/libmlir_c_runner_utils.so" \
    --shared-libs="$MLIR_AIR_LIB_DIR/libairgpu.so" \
    "$TMPDIR/s4.mlir"
