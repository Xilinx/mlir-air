#!/usr/bin/env bash
# THE b=0 TEST: can block 8 reach the memory bound block 1 already sits on?
#
# Block 1 runs at 44.2 GB/s and a layer measures 0.827 ms against an 0.826 ms
# weight-streaming floor. Extra tokens in a block add ZERO weight bytes, so if
# nothing else got in the way block 8 would cost what block 1 costs -- b = 0.
# It does not: b is 17.2 ms per token.
#
# This deletes everything that scales with the block and leaves the weight path
# untouched, on the workload that has the least else in it:
#
#   RMS_CHUNK_PROBE=2   the rms core's per-chunk X regeneration
#   PROJ_MM_PROBE=1     the projection mmul and the q4k unpack
#   vocab-only stream   no attention, no KV, no rope, no GLU
#
# Every channel, lock, descriptor and DMA stays where it was; the weights are
# still streamed, they are just not looked at. What is left IS the weight path.
#
#   lands near block 1's 7.278 ms -> the weight path is fine at block 8 and
#                                   everything blocking b=0 is now named
#   stays well above             -> there is a third term, and it is in the
#                                   transport rather than in either core
#
# The rms core is knowingly left broken here (stale X). This measures time, not
# correctness -- nothing it produces is meant to be read.
set -euo pipefail
cd "$(dirname "$0")"
export DECODE_MODEL=qwen3-4b-draft VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 \
       DECODE_NO_LM_WAVES=0 DECODE_HIDDEN_TAPS=0 DECODE_MASK_BIDIR=1 \
       RMS_CHUNK_PROBE=2 PROJ_MM_PROBE=1
unset DECODE_EXTRA_WAVES || true
DST=../llms/qwen3_4b_q4nx
L=${1:-511}
UDEC=5; UALL=15

for B in "${@:2}"; do
  [ "$B" = 8 ] && export DECODE_STACK=6080 || unset DECODE_STACK
  for LL in $L $((L+1)); do
    echo "=== z_b${B}_L${LL}  (full, waves [0,$UALL)) ==="; date
    rm -f decode_b${B}_L${LL}.xclbin decode_b${B}_L${LL}.insts.bin \
          "$DST/z_b${B}_L${LL}.xclbin" "$DST/z_b${B}_L${LL}.insts.bin"
    ./build_template.sh $B $LL
    mv -f decode_b${B}_L${LL}.xclbin    "$DST/z_b${B}_L${LL}.xclbin"
    mv -f decode_b${B}_L${LL}.insts.bin "$DST/z_b${B}_L${LL}.insts.bin"
  done
  for s in nolm lmonly; do
    case $s in
      nolm)   LO=0;     HI=$UDEC ;;
      lmonly) LO=$UDEC; HI=$UALL ;;
    esac
    echo "=== _z${s}_b${B}_L${L}.insts.bin  waves [$LO,$HI) ==="; date
    rm -f decode_b${B}_L${L}.xclbin decode_b${B}_L${L}.insts.bin \
          "$DST/_z${s}_b${B}_L${L}.insts.bin"
    UNI_WAVE_LO=$LO UNI_WAVE_HI=$HI ./build_template.sh $B $L
    mv -f decode_b${B}_L${L}.insts.bin "$DST/_z${s}_b${B}_L${L}.insts.bin"
    rm -f decode_b${B}_L${L}.xclbin
  done
done
unset RMS_CHUNK_PROBE PROJ_MM_PROBE
./build_template.sh 1 16 >/dev/null
date; echo "ZERO-WORK TEMPLATES DONE"
