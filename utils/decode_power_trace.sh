#!/usr/bin/env bash
# Sample package power / temp at 10 Hz while the q4nx profile runs.
OUT=${OUT:-/tmp/decode_trace}; mkdir -p "$OUT"
for h in /sys/class/hwmon/hwmon*; do
  case "$(cat $h/name 2>/dev/null)" in
    amdgpu)  PWR=$h/power1_input ;;
    k10temp) TCTL=$h/temp1_input ;;
  esac
done
GPUDEV=$(ls -d /sys/class/drm/card*/device 2>/dev/null | head -1)
( while :; do echo "$(date +%s%3N) $(cat $PWR) $(cat $TCTL) $(sed -n 's/.*: \([0-9]*\)Mhz \*/\1/p' $GPUDEV/pp_dpm_fclk|head -1)"; sleep 0.1; done ) > "$OUT/trace.log" &
SPID=$!
"$@" 2>&1 | tee "$OUT/run.log" | grep -E "Warm time to first token|Generated .* tokens|profile\] decode"
kill $SPID 2>/dev/null
python3 - "$OUT/trace.log" "$OUT/run.log" <<'PY'
import sys, re
rows=[l.split() for l in open(sys.argv[1]) if l.strip()]
w=[int(r[1])/1e6 for r in rows]; t=[int(r[2])/1000 for r in rows]
n=len(w)
if n:
    idle=sorted(w)[:max(1,n//20)]
    print(f"\npackage power over {n/10:.0f}s: min {min(w):.1f} W | median {sorted(w)[n//2]:.1f} W | p95 {sorted(w)[int(n*.95)]:.1f} W | max {max(w):.1f} W")
    print(f"peak Tctl {max(t):.0f} C")
    # busiest contiguous window = the decode phase
    top=sorted(range(n), key=lambda i:-w[i])[:max(1,n//5)]
    print(f"mean of busiest 20% of samples: {sum(w[i] for i in top)/len(top):.1f} W  <-- NPU-active package draw")
PY
