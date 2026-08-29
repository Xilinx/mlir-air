#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Time the dispatch across BATCH at a fixed context length.
#
# WHY THIS SEPARATES THE TWO STREAMS. Per q4k block a projection core pulls a
# weight block on one S2MM port and an activation tile on the other. The weight
# block is 2560 bf16 AT EVERY BATCH -- it is the same packed weights. The
# activation tile is BATCH*COL_BLOCK, so it is the only thing on that link that
# scales. Arithmetic scales too, but PROJ_MM_PROBE has already priced it. So the
# slope in BATCH is X traffic plus arithmetic, and the intercept is the weight
# stream and everything else that does not care how many tokens are in flight.
#
#   ./_batch_sweep.sh 161 8 16 32
#
# Batch 1 is NOT on this line: it runs the GEMV path, not the mmul path.
# Batches not divisible by 8 do not build -- proj_qmm_mm_flush_row de-tiles for
# aie::mmul<8,8,8>.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
L=${1:?usage: _batch_sweep.sh <L> <batch> [batch...]}
shift
for B in "$@"; do
  OUT="decode_b${B}_L${L}"
  # ALWAYS REBUILD. A template of the right NAME is not a template of the right
  # PROVENANCE: the probe sweeps in this directory leave decode_b8_L<L> behind
  # holding whatever PROJ_MM_PROBE they last built, and a reuse-if-exists check
  # would time that one and label it batch 8.
  echo ">>> building batch $B"
  rm -f "$OUT.xclbin" "$OUT.insts.bin"
  if ! env DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 \
       DECODE_STACK=6080 ./build_template.sh "$B" "$L" \
       > "/tmp/_batch_${B}.log" 2>&1; then
    echo "batch $B  BUILD FAILED (/tmp/_batch_${B}.log)"; continue
  fi
  env DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 DECODE_STACK=6080 \
    LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1 UNIFIED=1 DECODE_NO_LM_WAVES=1 \
    DECODE_BATCH="$B" DECODE_GOLDEN_L="$L" \
    "$PYTHON" -u _dispatch_probe.py "$OUT" 40000 9 2>/dev/null \
    | sed -n "s/^$OUT: /batch $B  /p"
done
