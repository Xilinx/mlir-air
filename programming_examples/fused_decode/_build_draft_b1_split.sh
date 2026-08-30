#!/usr/bin/env bash
# THE BATCHED PENALTY, IN THE SIMPLEST WORKLOAD THE ENGINE HAS.
#
# The open number (docs/DFlashFeasibility.md, "the batched layer is dataflow
# bound") is that a batch-8 layer runs 3.65x its weight-streaming floor with the
# arithmetic deleted, while the same design at batch 1 sits ON the floor. Every
# probe of it so far has used a whole decode LAYER -- attention, rope, KV, GLU,
# the projection and the rms core all at once.
#
# A VOCAB WAVE IS THE SAME PENALTY WITH ALMOST NOTHING IN IT. It has no
# attention, no KV cache, no rope and no GLU: it is a weight feed, a projection
# and an rms pass. At batch 8 it already measures 234.4 MB in 14.07 ms
# (17.5 GB/s) against batch 1's whole-model 44.2 GB/s -- so the penalty is
# present in a workload that has none of the things a layer has and a vocab wave
# does not.
#
# This builds the batch-1 side of that comparison so the ratio is measured on
# ONE dataflow at TWO batches, rather than inferred from a whole-model rate.
#
#   draft_b1_L<L>.xclbin/.insts.bin   the full 15-wave sequence at batch 1
#   _dlmonly_b1_L<L>.insts.bin        waves [5,15) -- the vocab path alone
#
# Same UNI_WAVE_LO/HI rule as everywhere else: the narrow build is the SAME
# device, so its insts.bin dispatches against the full xclbin.
set -euo pipefail
cd "$(dirname "$0")"
export DECODE_MODEL=qwen3-4b-draft VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 \
       DECODE_NO_LM_WAVES=0 DECODE_HIDDEN_TAPS=0 DECODE_MASK_BIDIR=1
unset DECODE_EXTRA_WAVES DECODE_STACK || true
DST=../llms/qwen3_4b_q4nx
L=${1:?usage: _build_draft_b1_split.sh <L>}
UDEC=5; UALL=15

echo "=== draft_b1_L$L  (full, waves [0,$UALL)) ==="; date
rm -f decode_b1_L$L.xclbin decode_b1_L$L.insts.bin \
      "$DST/draft_b1_L$L.xclbin" "$DST/draft_b1_L$L.insts.bin"
./build_template.sh 1 $L
mv -f decode_b1_L$L.xclbin   "$DST/draft_b1_L$L.xclbin"
mv -f decode_b1_L$L.insts.bin "$DST/draft_b1_L$L.insts.bin"

echo "=== _dlmonly_b1_L$L.insts.bin  waves [$UDEC,$UALL) ==="; date
rm -f decode_b1_L$L.xclbin decode_b1_L$L.insts.bin "$DST/_dlmonly_b1_L$L.insts.bin"
UNI_WAVE_LO=$UDEC UNI_WAVE_HI=$UALL ./build_template.sh 1 $L
mv -f decode_b1_L$L.insts.bin "$DST/_dlmonly_b1_L$L.insts.bin"
rm -f decode_b1_L$L.xclbin

echo "=== _dnolm_b1_L$L.insts.bin  waves [0,$UDEC) ==="; date
rm -f decode_b1_L$L.xclbin decode_b1_L$L.insts.bin "$DST/_dnolm_b1_L$L.insts.bin"
UNI_WAVE_LO=0 UNI_WAVE_HI=$UDEC ./build_template.sh 1 $L
mv -f decode_b1_L$L.insts.bin "$DST/_dnolm_b1_L$L.insts.bin"
rm -f decode_b1_L$L.xclbin

./build_template.sh 1 16 >/dev/null
date; echo "DRAFT B1 SPLIT DONE"
