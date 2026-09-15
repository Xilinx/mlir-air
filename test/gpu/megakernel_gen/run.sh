#!/usr/bin/env bash
#===- run.sh ---------------------------------------*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===------------------------------------------------------------------===//
#
# Generate a megakernel decode chain and run it. LAYERS controls how long the
# chain is; the point of the generator is that this number is not bounded by
# what anyone is willing to type.
#
#   LAYERS=8 ./run.sh
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR="${TMPDIR:-/tmp/air_megakernel_gen}"
LAYERS="${LAYERS:-4}"
DIM="${DIM:-128}"
REPEAT="${REPEAT:-1}"
TASKS="${TASKS:-8}"
WORKERS="${WORKERS:-32}"
TOKENS="${TOKENS:-1}"
mkdir -p "$TMPDIR"

if [ -z "${GFX_TARGET:-}" ]; then
  AMDGPU_ARCH_BIN=$(command -v amdgpu-arch || echo /opt/rocm/llvm/bin/amdgpu-arch)
  GFX_TARGET=$("$AMDGPU_ARCH_BIN" 2>/dev/null | head -1 | cut -d: -f1 || true)
fi
[ -n "$GFX_TARGET" ] || { echo "ERROR: set GFX_TARGET, e.g. GFX_TARGET=gfx942 $0" >&2; exit 1; }
echo "GFX_TARGET=$GFX_TARGET LAYERS=$LAYERS DIM=$DIM TASKS=$TASKS WORKERS=$WORKERS REPEAT=$REPEAT TOKENS=$TOKENS"

python3 "$SCRIPT_DIR/gen.py" --layers "$LAYERS" --dim "$DIM" --tasks "$TASKS" --workers "$WORKERS" --repeat "$REPEAT" --tokens "$TOKENS" > "$TMPDIR/chain.mlir"
air-opt "$TMPDIR/chain.mlir" -air-to-rocdl -o "$TMPDIR/s1.mlir"
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
