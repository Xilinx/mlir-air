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

# Cross-rank symmetric-heap test fundamentally requires a producer + a
# consumer process. Refuse single-process launches loudly rather than
# letting the kernel silently no-op or hang.
if [ "$NUM_RANKS" -lt 2 ]; then
  echo "ERROR: NUM_RANKS=$NUM_RANKS; this test requires >= 2 ranks (producer + consumer)." >&2
  exit 1
fi

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

# Input MLIR can be selected via INPUT env var.
#   atomic    — kernel-driven producer/consumer, LLVM atomicrmw + atomic
#               load with syncscope("") (Phase 2)
#   cacheline — kernel-driven producer/consumer, cache-line atomicity +
#               gpu.shuffle (Phase 2)
#   rank      — high-level air.rank form (Phase 3)
INPUT="${INPUT:-cacheline}"
case "$INPUT" in
  atomic|cacheline)
    # Kernel-driven test: needs the full GPU compilation chain
    # (rocdl-attach-target → convert-gpu-to-rocdl → gpu-module-to-binary).
    SRC_MLIR="$SCRIPT_DIR/air_sym_handwritten_${INPUT}.mlir"
    echo "Step 1a: Expand air.translate ops ($INPUT variant)"
    air-opt "$SRC_MLIR" --air-translate-to-llvm \
        -o "$TMPDIR/sym_post_translate.mlir"
    echo "Step 1b: Compile gpu.module to AMDGPU binary + finalize host"
    mlir-opt "$TMPDIR/sym_post_translate.mlir" \
        --pass-pipeline='builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary,func.func(gpu-async-region,convert-scf-to-cf),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)' \
        -o "$TMPDIR/sym_lowered.mlir"
    ;;
  rank)
    # Host-orchestrated test: simple LLVM-only pipeline.
    echo "Step 1a: Lower air.rank to mgpu*"
    air-opt "$SCRIPT_DIR/air_sym_with_rank.mlir" -air-rank-to-mgpu \
        -o "$TMPDIR/post_rank.mlir"
    echo "Step 1b: Lower IR to LLVM dialect"
    mlir-opt "$TMPDIR/post_rank.mlir" \
        --pass-pipeline='builtin.module(func.func(convert-scf-to-cf),convert-to-llvm,reconcile-unrealized-casts)' \
        -o "$TMPDIR/sym_lowered.mlir"
    ;;
  *)
    echo "Unknown INPUT=$INPUT; expected 'atomic', 'cacheline', or 'rank'" >&2; exit 1;;
esac

echo "Step 2: Run as ${NUM_RANKS} processes"
export AIRGPU_JOB_ID="${AIRGPU_JOB_ID:-$$}"

PIDS=()
PASS=1

for i in $(seq 0 $((NUM_RANKS - 1))); do
  (set -o pipefail
   # Pin each process to its own GPU at the OS / HIP-visibility level.
   # mlir-runner's built-in gpu.launch_func handler (and any nested call
   # into libmlir_rocm_runtime.so) only ever sees one device, so it can't
   # accidentally launch on the wrong one. Every rank still sees device 0
   # internally, so airgpu uses LOCAL_RANK=0.
   RANK=$i WORLD_SIZE=$NUM_RANKS LOCAL_RANK=0 HIP_VISIBLE_DEVICES=$i \
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
