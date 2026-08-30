#!/usr/bin/env bash
# IS THE RMS CORE'S CHUNK REGENERATION EXPOSED, OR OVERLAPPED?
#
# RMS_CHUNK_PROBE=2 (see _build_draft_rcp.sh) showed that deleting the
# regeneration takes the batch-8 vocab path from 13.800 ms to 9.091 -- 72% of
# the whole batch-8 penalty. It does NOT say whether that 4.71 ms is the
# kernel's arithmetic or the stall the kernel sits inside, and the two want
# opposite fixes: a faster rms_chunk (free) versus a second rstg staging buffer
# (8 KB of an L1 that is already 58.0 KB of 54).
#
# The counted arithmetic disagrees with the measurement by 6x. rms_chunk is
# hand-vectorised 16 wide and the file's own census puts it at 34 vector ops in
# 95 instructions -- about 0.75 cycles/element -- while 4.71 ms over the vocab
# path's 300 calls x 8 rows x 512 elements is 4.8 cycles/element.
#
# RMS_DELAY=<n> injects a scalar xorshift chain of n iterations at the top of
# rms_chunk_aie, once per regeneration, and touches nothing else. If wall time
# rises by the full injected amount and rises LINEARLY between two values of n,
# the rms core is on the critical path with no overlap at all -- and then the
# 4.71 ms is exposed compute and the cheap fix is the kernel. If the delay is
# absorbed, the core has slack, the 4.71 ms is stall, and only overlap helps.
#
# n must be large enough to clear the ~1.5 ms run-to-run spread: at ~4 cycles
# per xorshift iteration and 300 calls, n=5000 is about 5 ms.
#
#   RMSD=5000  bash _build_draft_rmsdelay.sh 511 8
#   RMSD=10000 bash _build_draft_rmsdelay.sh 511 8
set -euo pipefail
cd "$(dirname "$0")"
export DECODE_MODEL=qwen3-4b-draft VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 \
       DECODE_NO_LM_WAVES=0 DECODE_HIDDEN_TAPS=0 DECODE_MASK_BIDIR=1 \
       RMS_DELAY=${RMSD:?}
unset DECODE_EXTRA_WAVES || true
DST=../llms/qwen3_4b_q4nx
L=${1:-511}
UDEC=5; UALL=15

for B in "${@:2}"; do
  [ "$B" = 8 ] && export DECODE_STACK=6080 || unset DECODE_STACK
  for LL in $L $((L+1)); do
    echo "=== rmd${RMSD}_b${B}_L${LL}  (full, waves [0,$UALL)) ==="; date
    rm -f decode_b${B}_L${LL}.xclbin decode_b${B}_L${LL}.insts.bin \
          "$DST/rmd${RMSD}_b${B}_L${LL}.xclbin" "$DST/rmd${RMSD}_b${B}_L${LL}.insts.bin"
    ./build_template.sh $B $LL
    mv -f decode_b${B}_L${LL}.xclbin    "$DST/rmd${RMSD}_b${B}_L${LL}.xclbin"
    mv -f decode_b${B}_L${LL}.insts.bin "$DST/rmd${RMSD}_b${B}_L${LL}.insts.bin"
  done
  for s in nolm lmonly; do
    case $s in
      nolm)   LO=0;     HI=$UDEC ;;
      lmonly) LO=$UDEC; HI=$UALL ;;
    esac
    echo "=== _m${RMSD}${s}_b${B}_L${L}.insts.bin  waves [$LO,$HI) ==="; date
    rm -f decode_b${B}_L${L}.xclbin decode_b${B}_L${L}.insts.bin \
          "$DST/_m${RMSD}${s}_b${B}_L${L}.insts.bin"
    UNI_WAVE_LO=$LO UNI_WAVE_HI=$HI ./build_template.sh $B $L
    mv -f decode_b${B}_L${L}.insts.bin "$DST/_m${RMSD}${s}_b${B}_L${L}.insts.bin"
    rm -f decode_b${B}_L${L}.xclbin
  done
done
unset RMS_DELAY
./build_template.sh 1 16 >/dev/null
date; echo "RMS_DELAY TEMPLATES DONE"
