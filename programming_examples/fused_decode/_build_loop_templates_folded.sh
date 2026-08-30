#!/usr/bin/env bash
# The DFlash loop's templates, WITH the folded pre-pass in them.
#
# _build_loop_templates.sh's four builds, plus two extra instruction streams for
# the target's window. The target pair carries the 45-wave table but is built at
# UNI_WAVE_HI = UNI_DEC + UNI_LM + 25 -- everything through fc, and NOT the
# context K/V, whose X is target_hidden and cannot exist yet when a verify pass
# runs. The whole table has to be DECLARED either way or every wave behind the
# ones dropped would sit at a different offset in the extra weight BO.
#
# The extra streams are the same build at a narrower range. UNI_WAVE_LO/HI keep
# the ABI and the CDO fixed, so they produce the same device and a shorter
# insts.bin -- which is what lets the loop dispatch them against the shipping
# xclbin with no PDI switch. Their own xclbins are discarded.
set -euo pipefail
cd "$(dirname "$0")"
export DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 DECODE_STACK=6080 \
       DECODE_NO_LM_WAVES=0
DST=../llms/qwen3_4b_q4nx
W=${1:-128}
SW=${2:-$((W-1))}   # the stream L: DecodeInstsGen's base_L, the xclbin the loop registers

SPECS=$(cd $DST && "$PYTHON" -c "
import sys,json; sys.path.insert(0,'.')
import dflash_prepass_waves as P
w,_=P.wave_specs(P._load_draft_fd())
print(json.dumps([s.as_config() for s in w]))" 2>/dev/null | tail -1)
NW=$(echo "$SPECS" | tr -cd '{' | wc -c)
HI_V=$(cd $DST && "$PYTHON" -c "
import sys; sys.path.insert(0,'.')
import dflash_prepass_waves as P
w,_=P.wave_specs(P._load_draft_fd()); print(P.uni_hi_verify(w))" 2>/dev/null | tail -1)
HI_ALL=$(cd $DST && "$PYTHON" -c "
import sys; sys.path.insert(0,'.')
import dflash_prepass_waves as P
w,_=P.wave_specs(P._load_draft_fd())
print(P.uni_hi_verify(w) + sum(1 for s in w if s.group != 'fc'))" 2>/dev/null | tail -1)
LO_FC=$((HI_V - $(cd $DST && "$PYTHON" -c "
import sys; sys.path.insert(0,'.')
import dflash_prepass_waves as P
w,_=P.wave_specs(P._load_draft_fd()); print(sum(1 for s in w if s.group=='fc'))" 2>/dev/null | tail -1)))
echo ">>> $NW waves; verify [0,$HI_V)  fc [$LO_FC,$HI_V)  ctxkv [$HI_V,$HI_ALL)"

export DECODE_HIDDEN_TAPS=1 DECODE_MASK_BIDIR=0 DECODE_EXTRA_WAVES="$SPECS"
for L in $W $((W-1)); do
  echo "=== taps_b8_L$L  (verify stream) ==="; date
  rm -f taps_b8_L$L.xclbin taps_b8_L$L.insts.bin
  UNI_WAVE_LO=0 UNI_WAVE_HI=$HI_V ./build_template.sh 8 $L
  mv -f taps_b8_L$L.xclbin   $DST/
  mv -f taps_b8_L$L.insts.bin $DST/
done

# The two extra streams, at the ACTIVE TEMPLATE's L -- DecodeInstsGen picks the
# LOWEST L of a window as its base and registers THAT xclbin, so a stream named
# after the window would be dispatched against a device it was not built with.
# They carry no decode wave, so
# nothing in them depends on L either, but they must be the same build in every
# other respect as the xclbin they will be dispatched against.
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

# drafter, bidirectional, same window -- unchanged, no extra waves
unset DECODE_EXTRA_WAVES
export DECODE_HIDDEN_TAPS=0 DECODE_MASK_BIDIR=1 DECODE_MODEL=qwen3-4b-draft
for L in $W $((W-1)); do
  echo "=== draft_b8_L$L ==="; date
  rm -f decode_b8_L$L.xclbin decode_b8_L$L.insts.bin
  ./build_template.sh 8 $L
  mv -f decode_b8_L$L.xclbin    $DST/draft_b8_L$L.xclbin
  mv -f decode_b8_L$L.insts.bin $DST/draft_b8_L$L.insts.bin
done
./build_template.sh 1 16 >/dev/null
date; echo "ALL TEMPLATES DONE"
