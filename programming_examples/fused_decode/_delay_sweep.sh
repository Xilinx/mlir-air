#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Sweep PROJ_DELAY at a FIXED context length and time the dispatch. See
# kernels/proj_qmm.cc:250 -- this is the design's own slack probe, and it exists
# because the AIE trace unit does not work on this part. While the dispatch time
# is FLAT the proj core was idle at least that long per weight block and is not
# the critical path; past the knee it grows 1:1 and the slope calibrates the
# unit in cycles.
#
#   ./_delay_sweep.sh 8 161 0 64 256 1024
#
# L IS HELD FIXED AND IS NOT THE SWEEP AXIS. It is the context length: putting
# the delay in the template name via L makes every point a different amount of
# attention work, and the sweep then measures context scaling with a delay term
# hidden inside it. Every point overwrites the same template and is probed
# immediately, which is why this is a script and not four commands.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
B=${1:?usage: _delay_sweep.sh <batch> <L> <delay> [delay...]}
L=${2:?usage: _delay_sweep.sh <batch> <L> <delay> [delay...]}
shift 2
OUT="decode_b${B}_L${L}"

for D in "$@"; do
  echo ">>> building PROJ_DELAY=$D at L=$L"
  rm -f "$OUT.xclbin" "$OUT.insts.bin"
  if ! env ${D:+PROJ_DELAY=$D} DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 \
       W_DUAL_CHAN=1 DECODE_STACK=6080 \
       ./build_template.sh "$B" "$L" > "/tmp/_delay_${B}_${D}.log" 2>&1; then
    echo "    BUILD FAILED (see /tmp/_delay_${B}_${D}.log)"; continue
  fi
  env DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 DECODE_STACK=6080 \
    LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1 UNIFIED=1 DECODE_NO_LM_WAVES=1 \
    DECODE_BATCH="$B" DECODE_GOLDEN_L="$L" \
    "$PYTHON" -u _dispatch_probe.py "$OUT" 20000 9 2>/dev/null \
    | sed -n "s/^$OUT: /PROJ_DELAY=$D  /p"
done
