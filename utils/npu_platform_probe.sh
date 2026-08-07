#!/usr/bin/env bash
# Rootless platform probe for diagnosing NPU perf deltas between machines.
# No sudo, no dmidecode, no ryzenadj, no kernel modules. Drop into a CI step
# and diff the output against a known-good machine.
#
# Usage:  ./npu_platform_probe.sh [gemm_iters]     (default 20)

ITERS=${1:-20}
OUT=${OUT:-/tmp/npu_probe}; mkdir -p "$OUT"
GPUDEV=$(ls -d /sys/class/drm/card*/device 2>/dev/null | head -1)
PWR=""; TCTL=""
for h in /sys/class/hwmon/hwmon*; do
  case "$(cat $h/name 2>/dev/null)" in
    amdgpu)  [ -r "$h/power1_input" ] && PWR=$h/power1_input ;;
    k10temp) TCTL=$h/temp1_input ;;
  esac
done
hr(){ printf '\n===== %s =====\n' "$1"; }
now_ms(){ date +%s%3N; }

# 10 Hz background sampler of package power + CPU temp.
sampler(){ while :; do echo "$(now_ms) $(cat $PWR 2>/dev/null||echo 0) $(cat $TCTL 2>/dev/null||echo 0)"; sleep 0.1; done; }

hr "A1 platform identity"
for f in sys_vendor product_name board_name chassis_type; do
  printf '%-14s %s\n' "$f" "$(cat /sys/class/dmi/id/$f 2>/dev/null)"
done
echo "chassis_type key: 3=Desktop 9/10=Laptop/Notebook 35=MiniPC"

hr "A2 ACPI platform profile  <-- prime suspect"
echo "profile : $(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo 'NOT PRESENT (uncapped by OS)')"
echo "choices : $(cat /sys/firmware/acpi/platform_profile_choices 2>/dev/null || echo n/a)"

hr "A2b what privileges DO we have? (the workflow already uses 'sudo prlimit')"
echo "sudo -n -l:"; sudo -n -l 2>&1 | sed 's/^/  /' | head -20
PP=/sys/firmware/acpi/platform_profile
[ -e $PP ] && { [ -w $PP ] && echo "platform_profile is WRITABLE by $(id -un) -> can test 'performance' directly" \
                           || echo "platform_profile not writable by $(id -un); try: sudo -n tee $PP <<< performance"; }

hr "A3 CPU"
lscpu | grep -iE "^Model name|^CPU\(s\)|^Thread|max MHz|min MHz"
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "boost   : $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a)"

hr "A4 memory: DIMM count + SPD  <-- settles the bandwidth hypothesis"
python3 - <<'PY'
import glob, struct
e = sorted(glob.glob("/sys/bus/i2c/drivers/spd5118/*/eeprom"))
print(f"populated DDR5 DIMM slots (spd5118 hubs): {len(e)}")
if not e:
    print("  none -> soldered LPDDR5X (no SPD hub) or driver absent; rely on A5 mclk")
for p in e:
    b = open(p, "rb").read()
    mt   = {1:"RDIMM",2:"UDIMM",3:"SODIMM",4:"LRDIMM"}.get(b[3] & 0x0f, hex(b[3]))
    tck  = struct.unpack_from("<H", b, 20)[0]
    rate = round(2_000_000/tck/100)*100 if tck else 0
    dens = {0:0,1:4,2:8,3:12,4:16,5:24,6:32,7:48,8:64}.get(b[4] & 0x1f)
    rk   = ((b[234] >> 3) & 0x07) + 1
    print(f"  {p.split('/')[-2]}: DDR5 {mt} ~DDR5-{rate} tCK={tck}ps die={dens}Gb ranks={rk}")
PY
echo "MemTotal: $(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"

hr "A5 SoC clock ceilings (mclk=memory, fclk=fabric; * = active)"
for k in pp_dpm_mclk pp_dpm_fclk pp_dpm_socclk; do
  printf '%-14s %s\n' "$k" "$(cat $GPUDEV/$k 2>/dev/null | tr '\n' ' ')"
done

hr "A6 NPU / XRT"
xrt-smi examine 2>/dev/null | grep -iE "NPU Firmware|amdxdna Version|Processor"
xrt-smi examine -r platform 2>/dev/null | sed -n 's/.*\(Power Mode.*\)/\1/p'

hr "B sustained-vs-burst NPU: ${ITERS}x gemm validate  <-- THE decisive test"
[ -n "$PWR" ] && { sampler > "$OUT/pwr.log" & SPID=$!; }
: > "$OUT/gemm.log"
for i in $(seq 1 $ITERS); do
  t0=$(now_ms)
  tops=$(xrt-smi validate --run gemm 2>/dev/null | tr -d '\r' | sed -n 's/.*TOPS: *\([0-9.]*\).*/\1/p' | head -1)
  echo "$i $t0 $(now_ms) ${tops:-nan}" >> "$OUT/gemm.log"
done
[ -n "$SPID" ] && kill $SPID 2>/dev/null
python3 - "$OUT/gemm.log" "$OUT/pwr.log" <<'PY'
import sys
g = [l.split() for l in open(sys.argv[1])]
try:    p = [tuple(map(int, l.split())) for l in open(sys.argv[2])]
except Exception: p = []
print(f"{'iter':>4} {'TOPS':>6} {'secs':>6} {'pkg_W_peak':>11} {'Tctl_C':>7}")
tops = []
for i, t0, t1, v in g:
    w = [(pw, tc) for ts, pw, tc in p if int(t0) <= ts <= int(t1)]
    pk = max((x[0] for x in w), default=0)/1e6
    tc = max((x[1] for x in w), default=0)/1000
    print(f"{i:>4} {v:>6} {(int(t1)-int(t0))/1000:>6.1f} {pk:>11.1f} {tc:>7.0f}")
    try: tops.append(float(v))
    except ValueError: pass
if tops:
    n = max(1, len(tops)//4)
    print(f"\nfirst {n} mean = {sum(tops[:n])/n:.1f} TOPS | last {n} mean = {sum(tops[-n:])/n:.1f} TOPS")
    print("decay => sustained/thermal throttle | flat-low => static cap | flat-high => NPU is fine")
PY

hr "C host control: aggregate DRAM bandwidth + CPU GEMM"
python3 - <<'PY'
import multiprocessing as mp, time, numpy as np
def triad(_):
    N = 24 * 1024 * 1024                       # 192 MB/array, past any LLC
    a = np.ones(N); b = np.ones(N); c = np.empty(N)
    best = 0.0
    for _ in range(4):
        t = time.perf_counter(); np.multiply(b, 3.0, out=c); np.add(a, c, out=c)
        best = max(best, N * 24 / (time.perf_counter() - t) / 1e9)
    return best
for procs in (1, 4, 8):
    with mp.Pool(procs) as pool:
        print(f"STREAM-triad x{procs:<2} : {sum(pool.map(triad, range(procs))):6.1f} GB/s aggregate")
M = 4096
x = np.random.rand(M, M).astype(np.float32); y = np.random.rand(M, M).astype(np.float32)
best = 0.0
for _ in range(3):
    t = time.perf_counter(); x @ y; best = max(best, 2*M**3/(time.perf_counter()-t)/1e9)
print(f"sgemm {M}^3      : {best:6.1f} GFLOPS")
PY

hr "D sustained package power budget"
echo "idle    : $([ -n "$PWR" ] && awk '{printf "%.1f W", $1/1e6}' $PWR)  Tctl $([ -n "$TCTL" ] && awk '{printf "%.0f", $1/1000}' $TCTL)C"
for i in $(seq 1 $(nproc)); do (timeout 12 bash -c 'while :; do :; done') & done
sleep 4; echo "all-core@4s : $([ -n "$PWR" ] && awk '{printf "%.1f W", $1/1e6}' $PWR)  Tctl $([ -n "$TCTL" ] && awk '{printf "%.0f", $1/1000}' $TCTL)C"
sleep 6; echo "all-core@10s: $([ -n "$PWR" ] && awk '{printf "%.1f W", $1/1e6}' $PWR)  Tctl $([ -n "$TCTL" ] && awk '{printf "%.0f", $1/1000}' $TCTL)C  <-- sustained budget"
wait 2>/dev/null
echo; echo "probe complete"
