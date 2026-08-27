# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measure GEMM registry entries for shapes that are not in the JSON yet.

The registry (``details/GEMM_bf16_*.json``) stores, per (M, K, N) and per
method, the best measured tile config plus its throughput and accuracy.
``registry_lookup.gemm_config`` deliberately refuses to guess: a shape that is
not in the JSON raises ``KeyError`` rather than falling back to a neighbouring
config. So adding a new prefill length means measuring its shapes, and this is
the driver that does it -- previously a manual loop over ``run.py``.

## Seeding, and why it makes this cheap

The tile config is constrained by

    M % (tile_m * herd_m) == 0      K % tile_k_l2 == 0
    N % (tile_n * herd_n) == 0      tile_k_l2 % tile_k_l1 == 0

plus what fits L1. Only the first involves M -- so when the new M is a MULTIPLE
of an already-measured M (4096 from 2048, say), every config legal there is
still legal here, and the winner is a strong prior: L1 capacity, which is what
actually decides the tiling, does not change with M.

So the default is `--mode seed`: carry each method's winning config over from
`--seed-m` and measure it at the new M. One build per (shape, method). Use
`--mode neighbourhood` to also try the adjacent tile_m/tile_n powers of two,
which is worth it for the few largest shapes but multiplies the build count.

Compute is not the cost -- one pass over the 43 shapes LFM2/Qwen/Llama need at
M=4096 is ~6 TFLOP, about a second of NPU time. Every minute is `run.py`
building an xclbin, which is why this resumes rather than restarting.

## Usage

    # what would run, and how many builds
    python3 sweep_gemm.py --target-m 4096 --seed-m 2048 --dry-run

    # measure (resumable; re-running skips completed configs)
    python3 sweep_gemm.py --target-m 4096 --seed-m 2048 --out /tmp/m4096.json

    # emit registry-shaped entries to paste into details/*.json
    python3 sweep_gemm.py --target-m 4096 --emit /tmp/m4096.json

Results land in `--out` as they complete, so a killed run loses at most the
config in flight.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DETAILS = _HERE / "details"
_PROG = _HERE.parent

# output_dtype -> (registry JSON, harness directory)
_KERNELS = {
    "bf16": ("GEMM_bf16_in_bf16_out.json", "matrix_multiplication/bf16_in_bf16_out"),
    "f32": ("GEMM_bf16_in_fp32_out.json", "matrix_multiplication/bf16_in_fp32_out"),
}

# run.py's --high-precision for each method. "direct" is the low-precision tier
# (per-L2-tile truncation); the other two are FP32-accumulate.
_HIGH_PRECISION = {"fused-cast": "true", "drain": "true", "direct": "false"}

_RE_GFLOPS = re.compile(r"Throughput:\s*([0-9.eE+-]+)\s*GFLOP/s")
_RE_RELL1 = re.compile(r"mean_rel_L1=([0-9.eE+-]+)")


def load_registry(dtype):
    return json.loads((_DETAILS / _KERNELS[dtype][0]).read_text())


def shapes_at(reg, m):
    return {(s["K"], s["N"]): s for s in reg["shapes"] if s["M"] == m}


def _neighbours(tile):
    """Adjacent powers of two on tile_m and tile_n, the two that move most."""
    out = [dict(tile)]
    for key, lo, hi in (("tile_m", 32, 128), ("tile_n", 32, 256)):
        for f in (0.5, 2.0):
            v = int(tile[key] * f)
            if lo <= v <= hi:
                cand = dict(tile)
                cand[key] = v
                if cand not in out:
                    out.append(cand)
    return out


def plan(reg, target_m, seed_m, mode):
    """[(K, N, method, tile, used_by), ...] for every shape missing at target_m."""
    seeds, have = shapes_at(reg, seed_m), shapes_at(reg, target_m)
    jobs = []
    for (k, n), entry in sorted(seeds.items()):
        if (k, n) in have:
            continue
        for method, m in entry["methods"].items():
            tiles = [m["tile"]] if mode == "seed" else _neighbours(m["tile"])
            for t in tiles:
                jobs.append((k, n, method, t, entry.get("used_by", "")))
    return jobs


def legal(m, k, n, tile, herd):
    hm, hn = herd
    return (
        m % (tile["tile_m"] * hm) == 0
        and n % (tile["tile_n"] * hn) == 0
        and k % tile["tile_k_l2"] == 0
        and tile["tile_k_l2"] % tile["tile_k_l1"] == 0
    )


def run_one(harness, m, k, n, method, tile, herd, perf_iters, timeout, lock):
    """Build + run one config. Returns (ok, gflops, mean_rel_L1, seconds, note).

    Drives `make run`, NOT `run.py` directly. The harness compiles its AIE
    kernel object per tile config (`mm.o` is built with -DDIM_M=$(TILE_M)
    -DDIM_N=$(TILE_N) -DDIM_K=$(TILE_K_L1)) and runs from $(BUILD_DIR); calling
    run.py alone skips that and every config dies in ld.lld with
    "unable to find air_project/mm.o".
    """
    cmd = [
        "make",
        "run",
        f"M={m}",
        f"K={k}",
        f"N={n}",
        f"TILE_M={tile['tile_m']}",
        f"TILE_K_L2={tile['tile_k_l2']}",
        f"TILE_K_L1={tile['tile_k_l1']}",
        f"TILE_N={tile['tile_n']}",
        f"HERD_M={herd[0]}",
        f"HERD_N={herd[1]}",
        f"HIGH_PRECISION={_HIGH_PRECISION[method]}",
        f"PERF_ITERS={perf_iters}",
    ]
    # METHOD is a high-precision-only knob; the low-precision path rejects it.
    if _HIGH_PRECISION[method] == "true":
        cmd.append(f"METHOD={method}")
    # Take the shared NPU lock PER CONFIG rather than around the whole sweep.
    # The lock still spans this entire `make run`, compilation included -- the
    # harness builds and runs in one invocation, so this does NOT let anyone
    # else compile while we do. What it buys is a release point between every
    # config: another user waits for the config in flight, not for the hours
    # this sweep takes end to end.
    if lock:
        cmd = ["flock", "-x", "-w", str(timeout), lock] + cmd
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, cwd=harness, capture_output=True, text=True, timeout=timeout
        )
        out = p.stdout + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired:
        return False, None, None, time.time() - t0, "timeout"
    dt = time.time() - t0
    g = _RE_GFLOPS.search(out)
    r = _RE_RELL1.search(out)
    if rc != 0:
        # A build that cannot fit L1 is an expected prune, not an error.
        why = "build/run failed" if g is None else "numerical mismatch"
        return False, None, None, dt, why
    if g is None or r is None:
        # rc==0 but the harness printed no throughput or no precision line --
        # usually --perf-iters 0, or its output format drifted. Recording this
        # as a success would put a metric-less row in the results file, which
        # resume would then treat as measured and emit would silently drop.
        missing = " and ".join(
            n for n, m in (("throughput", g), ("precision", r)) if m is None
        )
        return False, None, None, dt, f"no {missing} line in harness output"
    return True, float(g.group(1)), float(r.group(1)), dt, ""


def emit(results, target_m, reg):
    """Fold measured configs into registry-shaped shape entries."""
    by_shape = {}
    for rec in results:
        # The gflops guard is redundant for rows this driver writes now, but a
        # results file from before that fix can still hold ok-but-metric-less rows.
        if not rec["ok"] or rec["gflops"] is None:
            continue
        key = (rec["K"], rec["N"])
        cur = by_shape.setdefault(
            key,
            {
                "M": target_m,
                "K": rec["K"],
                "N": rec["N"],
                "used_by": rec["used_by"],
                "methods": {},
                "best": {},
            },
        )
        prev = cur["methods"].get(rec["method"])
        if prev is None or rec["gflops"] > prev["gflops"]:
            cur["methods"][rec["method"]] = {
                "tile": rec["tile"],
                "gflops": round(rec["gflops"]),
                "mean_rel_L1": (
                    float(f"{rec['mean_rel_L1']:.2g}")
                    if rec["mean_rel_L1"] is not None
                    else None
                ),
                "tier": "low" if rec["method"] == "direct" else "high",
            }
    for entry in by_shape.values():
        for tier in ("high", "low"):
            cands = {m: v for m, v in entry["methods"].items() if v["tier"] == tier}
            if cands:
                entry["best"][tier] = max(cands, key=lambda m: cands[m]["gflops"])
    return sorted(by_shape.values(), key=lambda e: (e["K"], e["N"]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--target-m", type=int, required=True, help="M to measure (e.g. 4096)"
    )
    ap.add_argument(
        "--seed-m",
        type=int,
        default=2048,
        help="already-measured M to carry tile configs over from",
    )
    ap.add_argument("--dtype", choices=sorted(_KERNELS), default="bf16")
    ap.add_argument("--mode", choices=["seed", "neighbourhood"], default="seed")
    ap.add_argument("--out", default="", help="results JSON (resumable)")
    ap.add_argument(
        "--emit", default="", help="read a results JSON and print registry entries"
    )
    ap.add_argument("--perf-iters", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=1800, help="per config, seconds")
    ap.add_argument(
        "--filter", default="", help="only shapes whose used_by matches this substring"
    )
    ap.add_argument(
        "--lock",
        default="/tmp/mlir-air-npu.lock",
        help="flock file serializing NPU access, taken per config; empty to disable",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    reg = load_registry(a.dtype)
    herd = reg.get("herd", [8, 4])

    if a.emit:
        results = json.loads(Path(a.emit).read_text())
        print(json.dumps(emit(results, a.target_m, reg), indent=2))
        return 0

    jobs = plan(reg, a.target_m, a.seed_m, a.mode)
    if a.filter:
        jobs = [j for j in jobs if a.filter.lower() in j[4].lower()]
    if a.target_m % a.seed_m:
        print(
            f"WARNING: target M {a.target_m} is not a multiple of seed M "
            f"{a.seed_m}; seeded tile_m values may be illegal and get pruned.",
            file=sys.stderr,
        )

    skipped = [j for j in jobs if not legal(a.target_m, j[0], j[1], j[3], herd)]
    jobs = [j for j in jobs if legal(a.target_m, j[0], j[1], j[3], herd)]
    n_shapes = len({(k, n) for k, n, *_ in jobs})
    print(
        f"{len(jobs)} configs over {n_shapes} shapes "
        f"(mode={a.mode}, seed M={a.seed_m} -> target M={a.target_m})"
    )
    if skipped:
        print(
            f"  {len(skipped)} seeded configs are illegal at M={a.target_m} and were dropped"
        )

    out_path = Path(a.out) if a.out else None
    results = []
    if out_path and out_path.exists():
        results = json.loads(out_path.read_text())
        done = {
            (r["K"], r["N"], r["method"], json.dumps(r["tile"], sort_keys=True))
            for r in results
        }
        before = len(jobs)
        jobs = [
            j
            for j in jobs
            if (j[0], j[1], j[2], json.dumps(j[3], sort_keys=True)) not in done
        ]
        print(f"  resuming: {before - len(jobs)} already measured, {len(jobs)} left")

    if a.dry_run:
        for k, n, method, tile, used in jobs[:10]:
            print(f"  M{a.target_m} K{k} N{n:<6} {method:<11} {tile}  [{used[:40]}]")
        if len(jobs) > 10:
            print(f"  ... and {len(jobs) - 10} more")
        return 0

    harness = _PROG / _KERNELS[a.dtype][1]
    t_start = time.time()
    for i, (k, n, method, tile, used) in enumerate(jobs, 1):
        ok, g, rel, dt, note = run_one(
            harness,
            a.target_m,
            k,
            n,
            method,
            tile,
            herd,
            a.perf_iters,
            a.timeout,
            a.lock,
        )
        results.append(
            {
                "M": a.target_m,
                "K": k,
                "N": n,
                "method": method,
                "tile": tile,
                "used_by": used,
                "ok": ok,
                "gflops": g,
                "mean_rel_L1": rel,
                "seconds": round(dt, 1),
                "note": note,
            }
        )
        if out_path:
            out_path.write_text(json.dumps(results, indent=2))
        done_s = time.time() - t_start
        eta = done_s / i * (len(jobs) - i)
        # ok now implies both metrics parsed, so neither format can see None.
        status = f"{g:8.0f} GFLOP/s rel={rel:.1e}" if ok else f"SKIP ({note})"
        print(
            f"  [{i}/{len(jobs)}] K{k} N{n} {method:<11} {status}"
            f"  {dt:5.0f}s  eta {eta/60:.0f}m",
            flush=True,
        )
    print(
        f"\ndone in {(time.time() - t_start)/60:.0f} min; "
        f"{sum(1 for r in results if r['ok'])}/{len(results)} usable"
    )
    if out_path:
        print(
            f"results: {out_path}\nnow: sweep_gemm.py --target-m {a.target_m} "
            f"--emit {out_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
