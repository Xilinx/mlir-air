#!/usr/bin/env bash
# Just the two extra instruction streams, at a given L. Split out of
# _build_loop_templates_folded.sh so a naming fix does not rebuild four
# templates that are already correct.
set -euo pipefail
cd "$(dirname "$0")"
export DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 DECODE_STACK=6080 \
       DECODE_NO_LM_WAVES=0 DECODE_HIDDEN_TAPS=1 DECODE_MASK_BIDIR=0
DST=../llms/qwen3_4b_q4nx
SW=${1:?usage: _build_streams_only.sh <L>}
Q() { (cd $DST && "$PYTHON" -c "
import sys; sys.path.insert(0,'.')
import dflash_prepass_waves as P
w,_=P.wave_specs(P._load_draft_fd())
$1" 2>/dev/null | tail -1); }
export DECODE_EXTRA_WAVES=$(Q "
import json; print(json.dumps([s.as_config() for s in w]))")
HI_V=$(Q "print(P.uni_hi_verify(w))")
N_FC=$(Q "print(sum(1 for s in w if s.group=='fc'))")
HI_ALL=$(Q "print(P.uni_hi_verify(w) + sum(1 for s in w if s.group != 'fc'))")
LO_FC=$((HI_V - N_FC))
for s in fc ctxkv; do
  case $s in
    fc)    LO=$LO_FC; HI=$HI_V ;;
    ctxkv) LO=$HI_V;  HI=$HI_ALL ;;
  esac
  echo "=== _L${SW}_$s.insts.bin  waves [$LO,$HI) ==="; date
  rm -f taps_b8_L$SW.xclbin taps_b8_L$SW.insts.bin
  UNI_WAVE_LO=$LO UNI_WAVE_HI=$HI ./build_template.sh 8 $SW
  mv -f taps_b8_L$SW.insts.bin $DST/_L${SW}_$s.insts.bin
  rm -f taps_b8_L$SW.xclbin
done
./build_template.sh 1 16 >/dev/null
date; echo "STREAMS DONE"
