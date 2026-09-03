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

# THE KERNELS MUST BE COMPILED FOR THE MODEL THE DESIGN IS BUILT FOR.
# makefile_kbase() returns THIS directory's Makefile PEANO_KBASE, which
# hardcodes -DMODEL_TYPE=LLAMA_3_2_1B: that Makefile only ever builds
# llama-3.2-1b, and each llms/<model>_q4nx/Makefile carries its own
# -DMODEL_TYPE. Taken verbatim, every DECODE_MODEL!=llama-3.2-1b template this
# script produced had LLAMA kernels (MODEL_DIM 2048, DH 64) inside a design
# built for another model's dimensions -- and the failure is silent: it links,
# it dispatches, it returns COMPLETED, and the layer output is simply never
# written (measured on qwen3-4b: X came back bit-identical to the host fill
# with the tail zeroed, at batch 1, where the production template writes all
# of it). Substitute the model's own MODEL_TYPE here so the object and the
# design cannot disagree -- the same rule this script already applies to
# PROJ_MM_BATCH.
case "$DECODE_MODEL" in
  llama-3.2-1b)          MT=LLAMA_3_2_1B ;;
  llama-3.2-3b)          MT=LLAMA_3_2_3B ;;
  llama-3.1-8b)          MT=LLAMA_3_1_8B ;;
  gemma3-4b)             MT=GEMMA3_4B ;;
  phi4-mini)             MT=PHI4_4B ;;
  qwen2.5-3b)            MT=QWEN2_5_3B ;;
  qwen2.5-7b)            MT=QWEN2_5_7B ;;
  qwen3-8b)              MT=QWEN3_8B ;;
  # The DFlash drafter is qwen3-4b's per-layer geometry with fewer layers, so
  # it takes qwen3-4b's kernels unchanged; only UNI_DEC differs, and that is a
  # builder-side constant, not a kernel one.
  qwen3-4b|qwen3-4b-draft) MT=QWEN3_4B ;;
  lfm2-1.2b)             MT=LFM2_1_2B ;;
  *) echo "build_template.sh: no MODEL_TYPE known for DECODE_MODEL=$DECODE_MODEL" >&2
     echo "  add it to the case above AND to models/all_models.h" >&2; exit 1 ;;
esac
KB=$(echo "$KB" | sed "s/-DMODEL_TYPE=[A-Z0-9_]*/-DMODEL_TYPE=$MT/")
echo ">>> kernels for MODEL_TYPE=$MT  [DECODE_MODEL=$DECODE_MODEL]"

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
  # RMS_MEMTILE_REFEED=3 needs the row-major producer entry point, which is
  # behind RMS_ROW_OUT so that a build not asking for it stays byte-identical.
  # `$(test && echo)` would exit 1 when the test is false, and the enclosing
  # assignment inherits that status -- under `set -e` every build that is NOT
  # mode 3 then dies here, silently, after printing "rms_residual.o". Use an
  # if/fi so the substitution always succeeds.
  [ "$k" = "rms_residual" ] && EXTRA="${RMS_CHUNK_PROBE:+-DRMS_CHUNK_PROBE=$RMS_CHUNK_PROBE} ${RMS_DELAY:+-DRMS_DELAY=$RMS_DELAY} $(if [ "${RMS_MEMTILE_REFEED:-0}" = 3 ]; then echo -DRMS_ROW_OUT; fi)"
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
# A dynseq build ships a .txn.h and NO insts.bin, so it cannot stand in for a
# frozen template -- and it would delete the insts.bin of whatever it overwrote.
# Its own prefix keeps it away from the templates the shipping gates read
# (batch_equiv.py --prefix dyn picks it back up).
[ "${DECODE_DYNSEQ:-0}" = "1" ] && PFX=dyn
# A traced build appends the trace region to the rms BO, so its ABI is wider
# than the template of the same name a numeric gate would bind. Same rule, same
# reason as the two above.
[ -n "${DECODE_TRACE_SIZE:-}" ] && [ "${DECODE_TRACE_SIZE:-0}" != "0" ] && PFX=trace
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
# DECODE_DYNSEQ=1 takes the context length at dispatch, so there is no frozen
# insts.bin to move: the compiler emits a TXN builder the host calls per L. It
# lands in air_project/ under a fixed name, so it has to be copied out beside
# the xclbin or the next build overwrites it -- the same rule that makes every
# other artifact here carry its batch and L in the filename.
if [ "${DECODE_DYNSEQ:-0}" = "1" ]; then
  if [ -f decode.xclbin ] && [ -f air_project/npu.air.txn.h ]; then
    mv -f decode.xclbin "$OUT.xclbin"
    cp -f air_project/npu.air.txn.h "$OUT.txn.h"
    rm -f decode.insts.bin
    echo "    $OUT.xclbin + $OUT.txn.h  [dynseq]"
    exit 0
  fi
  echo "    FAILED (dynseq) -- see $LOG"
  grep -m5 "error:" "$LOG" || tail -5 "$LOG"
  exit 1
fi
if [ -f decode.xclbin ] && [ -f decode.insts.bin ]; then
  mv -f decode.xclbin "$OUT.xclbin"
  mv -f decode.insts.bin "$OUT.insts.bin"
  echo "    $OUT.xclbin + $OUT.insts.bin"
else
  echo "    FAILED -- see $LOG"
  grep -m5 "error:" "$LOG" || tail -5 "$LOG"
  exit 1
fi
