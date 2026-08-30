#!/usr/bin/env bash
# THE BATCHED PENALTY AS A FUNCTION OF BATCH, on the minimal workload.
#
# docs/DFlashFeasibility.md 3.17 measured the vocab-only stream at 1.90x going
# from batch 1 to batch 8 on identical weights -- a weight feed, a projection and
# an rms pass, with no attention, KV, rope or GLU in it. Two endpoints do not
# say what SHAPE that is, and the shape is the whole diagnosis:
#
#   t(B) affine in B      -> a per-row cost serialized with the weight stream
#   t(B) flat above B=1   -> NOT about batch at all. build_template.sh compiles
#                            -DPROJ_MM_BATCH only when BATCH != 1, so batch 1
#                            runs the v1 GEMV and every other batch runs the q4k
#                            mmul. A step at 1->2 is a KERNEL PATH difference
#                            wearing a batch costume.
#   t(B) saturating       -> contention for a shared resource
#
# Builds the missing middle: batch 2 and 4, full pair plus both split streams,
# same L and same everything else as the batch-1 and batch-8 sets already on
# disk. Cheap -- the drafter is 5 layers.
set -euo pipefail
cd "$(dirname "$0")"
export DECODE_MODEL=qwen3-4b-draft VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 \
       DECODE_NO_LM_WAVES=0 DECODE_HIDDEN_TAPS=0 DECODE_MASK_BIDIR=1
unset DECODE_EXTRA_WAVES DECODE_STACK || true
DST=../llms/qwen3_4b_q4nx
L=${1:-511}
UDEC=5; UALL=15

for B in "${@:2}"; do
  # The pair: DecodeInstsGen fits its slope from two same-ATTN_MAXL builds and
  # refuses to open a family with only one, even for a stream that has no
  # L-dependence in it at all.
  for LL in $L $((L+1)); do
    echo "=== draft_b${B}_L${LL}  (full, waves [0,$UALL)) ==="; date
    rm -f decode_b${B}_L${LL}.xclbin decode_b${B}_L${LL}.insts.bin
    ./build_template.sh $B $LL
    mv -f decode_b${B}_L${LL}.xclbin   "$DST/draft_b${B}_L${LL}.xclbin"
    mv -f decode_b${B}_L${LL}.insts.bin "$DST/draft_b${B}_L${LL}.insts.bin"
  done
  for s in nolm lmonly; do
    case $s in
      nolm)   LO=0;     HI=$UDEC ;;
      lmonly) LO=$UDEC; HI=$UALL ;;
    esac
    echo "=== _d${s}_b${B}_L${L}.insts.bin  waves [$LO,$HI) ==="; date
    rm -f decode_b${B}_L${L}.xclbin decode_b${B}_L${L}.insts.bin
    UNI_WAVE_LO=$LO UNI_WAVE_HI=$HI ./build_template.sh $B $L
    mv -f decode_b${B}_L${L}.insts.bin "$DST/_d${s}_b${B}_L${L}.insts.bin"
    rm -f decode_b${B}_L${L}.xclbin
  done
done
./build_template.sh 1 16 >/dev/null
date; echo "BATCH SWEEP TEMPLATES DONE"
