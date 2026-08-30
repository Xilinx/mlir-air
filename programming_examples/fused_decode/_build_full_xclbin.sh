#!/usr/bin/env bash
# The TARGET's xclbin, built at the FULL wave table.
#
# The instruction streams are built at narrower UNI_WAVE ranges and dispatched
# against this one device. Measured at L=157: every stream is correct against an
# xclbin built at [0, ALL) and the fc stream reads cos 0.008 against one built at
# [0, verify) -- so the range truncates something in the DEVICE and not only in
# the stream. The rule that falls out: ONE xclbin at the whole table, N streams
# under it. This builds the xclbin; the .insts.bin it produces is discarded,
# because the loop's verify stream is the [0, verify) build's.
set -euo pipefail
cd "$(dirname "$0")"
export DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 DECODE_STACK=6080 \
       DECODE_NO_LM_WAVES=0 DECODE_HIDDEN_TAPS=1 DECODE_MASK_BIDIR=0
DST=../llms/qwen3_4b_q4nx
Q() { (cd $DST && "$PYTHON" -c "
import sys; sys.path.insert(0,'.')
import dflash_prepass_waves as P
w,_=P.wave_specs(P._load_draft_fd())
$1" 2>/dev/null | tail -1); }
export DECODE_EXTRA_WAVES=$(Q "
import json; print(json.dumps([s.as_config() for s in w]))")
HI_ALL=$(Q "print(P.uni_hi_verify(w) + sum(1 for s in w if s.group != 'fc'))")
for L in "$@"; do
  echo "=== taps_b8_L$L.xclbin  waves [0,$HI_ALL) ==="; date
  rm -f taps_b8_L$L.xclbin taps_b8_L$L.insts.bin
  UNI_WAVE_LO=0 UNI_WAVE_HI=$HI_ALL ./build_template.sh 8 $L
  mv -f taps_b8_L$L.xclbin $DST/taps_b8_L$L.xclbin
  rm -f taps_b8_L$L.insts.bin
done
./build_template.sh 1 16 >/dev/null
date; echo "XCLBINS DONE"
