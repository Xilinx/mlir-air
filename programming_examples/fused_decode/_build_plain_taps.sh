#!/usr/bin/env bash
# A batch-8 taps template at THIS HEAD with NO extra waves, under its own prefix.
#
# A CONTROL. The verify sweep degraded after the fold (agreement 8/8 everywhere
# -> 4..8/8, and corr at P=5 below the noise floor), and there are two candidate
# causes at this HEAD: the extra waves in the template, or everything else that
# moved since the baseline was taken (the rms_chunk unroll above all). Same
# builder, same kernels, same window -- only the wave table removed.
set -euo pipefail
cd "$(dirname "$0")"
export DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 DECODE_STACK=6080 \
       DECODE_NO_LM_WAVES=0 DECODE_HIDDEN_TAPS=1 DECODE_MASK_BIDIR=0
unset DECODE_EXTRA_WAVES UNI_WAVE_LO UNI_WAVE_HI
DST=../llms/qwen3_4b_q4nx
for L in "$@"; do
  echo "=== plain_b8_L$L (no extra waves) ==="; date
  rm -f taps_b8_L$L.xclbin taps_b8_L$L.insts.bin
  ./build_template.sh 8 "$L"
  mv -f taps_b8_L$L.xclbin    $DST/plain_b8_L$L.xclbin
  mv -f taps_b8_L$L.insts.bin $DST/plain_b8_L$L.insts.bin
done
./build_template.sh 1 16 >/dev/null
date; echo "PLAIN TAPS DONE"
