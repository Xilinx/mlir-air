#!/usr/bin/env bash
#===- run.sh - Multi-process symmetric-heap DMA e2e test --*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===------------------------------------------------------------------===//
#
# Compile and run the hand-written symmetric-heap MLIR test as N processes.
# Each process executes the full IR; processes coordinate via the symmetric
# heap (XGMI peer-mapped VMem buffers).
#
# Usage: run.sh [num_ranks]   (default: 2)
#
# Required environment (auto-detected when sourced via env_setup_gpu.sh):
#   MLIR_AIR_INSTALL_DIR  - path containing lib/libairgpu.so
#   LLVM_INSTALL_DIR      - path containing bin/mlir-opt + lib/libmlir_*.so
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_RANKS=${1:-2}
TMPDIR="${TMPDIR:-/tmp/air_sym_dma}"
mkdir -p "$TMPDIR"

# Refuse to run if there aren't enough physically distinct GPUs for one
# rank per GPU. Colocating ranks on a single GPU would make XGMI/peer-VA
# transparently fall back to local memory and produce false-positive PASSes.
if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
  NUM_GPUS=$(echo "$HIP_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
  NUM_GPUS=$(grep -l '^simd_count [1-9]' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | wc -l)
fi
if [ "$NUM_GPUS" -lt "$NUM_RANKS" ]; then
  echo "ERROR: need >= $NUM_RANKS GPUs to validate cross-rank XGMI traffic; found $NUM_GPUS." >&2
  echo "       This test refuses to colocate ranks on a single GPU because it would" >&2
  echo "       silently bypass the symmetric-heap path and report false PASSes." >&2
  exit 1
fi

LLVM_LIB_DIR="${LLVM_INSTALL_DIR:-$(dirname "$(which mlir-opt)")/..}/lib"
AIRGPU_LIB="${MLIR_AIR_INSTALL_DIR:-$(dirname "$(which air-opt)")/..}/lib/libairgpu.so"

echo "Step 1: Lower hand-written IR to LLVM dialect"
mlir-opt "$SCRIPT_DIR/air_sym_handwritten.mlir" \
    --pass-pipeline='builtin.module(func.func(convert-scf-to-cf),convert-to-llvm,reconcile-unrealized-casts)' \
    -o "$TMPDIR/sym_lowered.mlir"

echo "Step 2: Run as ${NUM_RANKS} processes"
export AIRGPU_JOB_ID="${AIRGPU_JOB_ID:-$$}"

PIDS=()
PASS=1

for i in $(seq 0 $((NUM_RANKS - 1))); do
  (set -o pipefail
   RANK=$i WORLD_SIZE=$NUM_RANKS LOCAL_RANK=$i \
   mlir-runner --entry-point-result=void \
       --shared-libs="$LLVM_LIB_DIR/libmlir_rocm_runtime.so" \
       --shared-libs="$AIRGPU_LIB" \
       --shared-libs="$LLVM_LIB_DIR/libmlir_runner_utils.so" \
       --shared-libs="$LLVM_LIB_DIR/libmlir_c_runner_utils.so" \
       "$TMPDIR/sym_lowered.mlir" 2>&1 | sed "s/^/[rank $i] /") &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    PASS=0
  fi
done

if [ $PASS -eq 1 ]; then
  echo "=== ALL ${NUM_RANKS} RANKS PASSED ==="
else
  echo "=== SOME RANKS FAILED ==="
  exit 1
fi
