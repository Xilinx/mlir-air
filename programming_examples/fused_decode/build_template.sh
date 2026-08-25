#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build ONE decode template at a given batch and context length, and name it
# after both. The Makefile's compile-decode builds the shipping PAIR (L and
# L-1, for the DecodeInstsGen slope) at batch 1; this is the single-template
# form the batching work needs, where the interesting axis is DECODE_BATCH.
#
# WHY A SCRIPT AND NOT A MAKE TARGET. It has to rebuild proj_qmm.o with
# -DPROJ_MM_BATCH=<batch> for every batch it builds, and a stale .o from the
# previous batch is not a build error -- it is `ld.lld: undefined symbol:
# proj_qmm_mm_zero` at best, and at worst a batch-8 design linked against
# batch-1 kernels. Rebuilding it here means the object and the template cannot
# disagree.
#
# The Peano PIN PREFLIGHT IS SKIPPED. It has to be, in a sandbox whose nightly
# index no longer carries the pinned build -- but the pin exists because a stale
# Peano miscompiles the inlined attention into "correct first token, then
# garbage". So: templates from this script are good for DATAFLOW work (does it
# deadlock, do the descriptors line up) and NOT for numerics claims.
#
#   ./build_template.sh 8 128            -> decode_b8_L128.{xclbin,insts.bin}
#   ./build_template.sh 1 129            -> decode_b1_L129.{xclbin,insts.bin}
#   DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 ./build_template.sh 8 2048
set -euo pipefail

BATCH=${1:?usage: build_template.sh <batch> <L>}
L=${2:?usage: build_template.sh <batch> <L>}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

: "${PYTHON:=python3}"
: "${PEANO_INSTALL_DIR:?PEANO_INSTALL_DIR must point at an llvm-aie install}"
: "${DECODE_MODEL:=llama-3.2-1b}"
: "${VOCAB_CHUNK_I2:=18}"

KB=$("$PYTHON" -c "import sys; sys.path.insert(0,'.');
import check_kernels_inert as C; print(' '.join(C.makefile_kbase()))")
CLANG="$PEANO_INSTALL_DIR/bin/clang++"

# proj_qmm's batched entry points are behind -DPROJ_MM_BATCH so that a build
# which does not ask for them does not even parse q4k_mm.h. Every other batched
# kernel takes the batch as a runtime argument.
MMDEF=""; ATTNDEF=""
if [ "$BATCH" != "1" ]; then
  MMDEF="-DPROJ_MM_BATCH=$BATCH"
  ATTNDEF="-DATTN_BATCH=$BATCH"
fi

echo ">>> kernels (batch $BATCH)"
for k in proj_qmm rms_residual glu rope; do
  echo "    $k.o"
  EXTRA=""; [ "$k" = "proj_qmm" ] && EXTRA="$MMDEF ${PROJ_FLUSH_PROBE:+-DPROJ_FLUSH_PROBE=$PROJ_FLUSH_PROBE} ${PROJ_MM_PROBE:+-DPROJ_MM_PROBE=$PROJ_MM_PROBE} ${PROJ_DELAY:+-DPROJ_DELAY=$PROJ_DELAY}"
  [ "$k" = "rms_residual" ] && EXTRA="${RMS_CHUNK_PROBE:+-DRMS_CHUNK_PROBE}"
  # GLU_ROW_PROBE=<n>: diagnostic variants of the batched GLU row (see glu.cc).
  # Batch-only, so the shipping kernel stays inert either way.
  [ "$k" = "glu" ] && [ -n "${GLU_ROW_PROBE:-}" ] && EXTRA="-DGLU_ROW_PROBE=$GLU_ROW_PROBE"
  # shellcheck disable=SC2086
  "$CLANG" $KB $EXTRA -O2 -c "kernels/$k.cc" -o "$k.o"
done
# -O1 and inline-only: attn at -O2 hits a do-while deadlock, rope at -O1
# miscompiles. Same split as the Makefile, for the same reasons.
for k in attn_qk attn_kv; do
  echo "    $k.ll"
  # shellcheck disable=SC2086
  "$CLANG" $KB $ATTNDEF -O1 -DDECODE_INLINE_ATTN -S -emit-llvm "kernels/$k.cc" -o "$k.ll"
done

# A diagnostic build has different BO sizes and extra shim tasks, so it must not
# overwrite the template a numeric run would pick up. Each gets its own prefix,
# and batch_equiv.py --prefix picks it back up:
#   DECODE_PROBE=1       python batch_equiv.py --prefix probe --batch B --L L --smoke
#   DECODE_HIDDEN_TAPS=1 python batch_equiv.py --prefix taps  --batch B --L L --smoke
PFX=decode
[ "${DECODE_PROBE:-0}" != "0" ] && PFX=probe
[ "${DECODE_HIDDEN_TAPS:-0}" != "0" ] && PFX=taps
OUT="${PFX}_b${BATCH}_L${L}"
echo ">>> template $OUT  [$DECODE_MODEL]"
LOG="${TMPDIR:-/tmp}/${OUT}.log"
# Decode-only by default: this script builds the template the BATCH-EQUIVALENCE
# gates read, and those read layer outputs, not logits. Seven vocab waves would be
# most of the build and most of the run for something they never look at. Set
# DECODE_NO_LM_WAVES=0 to build the full fused sequence (decode + LM head).
env VOCAB_CHUNK_I2="$VOCAB_CHUNK_I2" LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1 UNIFIED=1 \
    DECODE_NO_LM_WAVES="${DECODE_NO_LM_WAVES:-1}" \
    DECODE_MODEL="$DECODE_MODEL" DECODE_BATCH="$BATCH" DECODE_GOLDEN_L="$L" \
    "$PYTHON" fused_decode.py > "$LOG" 2>&1 || true
if [ -f decode.xclbin ] && [ -f decode.insts.bin ]; then
  mv -f decode.xclbin "$OUT.xclbin"
  mv -f decode.insts.bin "$OUT.insts.bin"
  echo "    $OUT.xclbin + $OUT.insts.bin"
else
  echo "    FAILED -- see $LOG"
  grep -m5 "error:" "$LOG" || tail -5 "$LOG"
  exit 1
fi
