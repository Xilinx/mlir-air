#!/usr/bin/env bash
# PRICE THE DRAFTER'S LM HEAD BY DELETION.
#
# The drafter runs the SAME 151936-wide vocab projection the target does, over
# 5 layers instead of 36 -- so it is a much larger fraction of a drafter
# dispatch than of a target one, and 157.5/36 = 4.4 ms per target layer against
# 37.5/5 = 7.5 ms per drafter layer says something is paying for it.
#
# UNI_WAVE_LO/HI are build-time and restrict ONLY the fused launch loop; the ABI
# and the CDO stay fixed. So these are the SAME device as draft_b8_L<L>.xclbin
# with a shorter insts.bin, dispatchable against it through
# FusedDecoder.dispatch_insts with no PDI switch. Their own xclbins are dropped.
#
#   _dnolm_L<L>.insts.bin   waves [0,UNI_DEC)   -- the 5 decode layers alone
#   _dlmonly_L<L>.insts.bin waves [UNI_DEC,ALL) -- the 10 vocab waves alone
#
# The pair is a control on itself: decode-only + lm-only should sum to the full
# stream, within the ~1.5 ms run-to-run spread.
set -euo pipefail
cd "$(dirname "$0")"
export DECODE_MODEL=qwen3-4b-draft VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 \
       DECODE_STACK=6080 DECODE_NO_LM_WAVES=0 DECODE_HIDDEN_TAPS=0 \
       DECODE_MASK_BIDIR=1
unset DECODE_EXTRA_WAVES || true
DST=../llms/qwen3_4b_q4nx
L=${1:?usage: _build_draft_split_streams.sh <L>}

read -r UDEC UALL < <("$PYTHON" -c "
import os,sys; sys.path.insert(0,'.')
os.environ.update(FUSED_DECODE_EMIT_ONLY='1', LM_HEAD='0', NLAYERS='1',
                  DECODE_GOLDEN='1', UNIFIED='1', DECODE_BATCH='8',
                  DECODE_GOLDEN_L='$L')
import importlib.util as I
s=I.spec_from_file_location('fd','fused_decode.py'); m=I.module_from_spec(s)
try: s.loader.exec_module(m)
except SystemExit: pass
print(m.UNI_DEC, m.UNI_DEC+m.UNI_LM)" 2>/dev/null | tail -1)
echo ">>> drafter UNI_DEC=$UDEC  UNI_WAVES=$UALL"

for s in nolm lmonly; do
  case $s in
    nolm)   LO=0;     HI=$UDEC ;;
    lmonly) LO=$UDEC; HI=$UALL ;;
  esac
  echo "=== _d${s}_L${L}.insts.bin  waves [$LO,$HI) ==="; date
  rm -f decode_b8_L$L.xclbin decode_b8_L$L.insts.bin "$DST/_d${s}_L${L}.insts.bin"
  UNI_WAVE_LO=$LO UNI_WAVE_HI=$HI ./build_template.sh 8 $L
  mv -f decode_b8_L$L.insts.bin "$DST/_d${s}_L${L}.insts.bin"
  rm -f decode_b8_L$L.xclbin
done
./build_template.sh 1 16 >/dev/null
date; echo "DRAFT SPLIT STREAMS DONE"
