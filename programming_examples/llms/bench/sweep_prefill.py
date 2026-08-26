# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Sweep prefill TTFT against padded prefill length into JSON.

The companion to sweep_decode.py, on the other axis. Decode throughput is a
function of KV depth; TTFT is a function of how many token rows the prefill
engines compute, so the two need different sweeps and cannot share a table.

The x-axis here is the PADDED prefill length (the length the engines are built
for), not the number of real prompt tokens. That distinction is the whole point
of this file: these prefills pad the prompt to the built length and read the
last real token's row, so at a fixed engine TTFT does not move with the prompt.
Measured on llama32_1b_q4nx at a 2048 engine: 959 / 965 / 978 / 983 ms for
128 / 512 / 1024 / 2048 real tokens -- 2.5% across a 16x prompt range, i.e. a
flat line. Rebuilding the engines per length gives the real curve: 315 / 524 /
981 ms at 512 / 1024 / 2048.

So every point here rebuilds (or reuses a per-length cache of) the prefill ELFs
and then times a warm prefill at that length. The runner GUARDS that the engines
were actually built at the requested length -- three models' Makefiles wire CTX
only to the bench length and not to the engine's seq_len, which silently
produces the flat curve above rather than an error.

Latency only, never a correctness gate -- `make verify` is the correctness gate.
The per-length Paris first-token check the prefill drivers already print is
recorded alongside each point, because a borrowed-tile or wrong-length build
tends to show up there first.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# "Time to first token (TTFT): 0.981s" -- the line every prefill driver prints
# for extract_perf.py, reused here so the two captures cannot disagree.
TTFT_RE = re.compile(r"^\s*Time to first token \(TTFT\):\s*([\d.]+)\s*s", re.M)
# "[bench] L=2048: WALL=981ms 2087 tok/s prefill  |  NPU-dispatch=953ms ..."
BENCH_RE = re.compile(
    r"^\[bench\] L=(\d+):\s*WALL=([\d.]+)ms.*?NPU-dispatch=([\d.]+)ms", re.M
)
# "[q4nx_prefill] constructing seq_len=2048 (compiling engines)..." (q4nx family)
# "Compiling LFM2 prefill kernels (seq_len=2048)"                   (lfm2)
SEQ_RE = re.compile(r"(?:constructing seq_len=|prefill kernels \(seq_len=)(\d+)")
PARIS_RE = re.compile(r"\*\*\* PARIS \*\*\*")
MISS_RE = re.compile(r"\]\s*MISS\b")

# Device-side "the dispatch did not complete" evidence. Same list sweep_decode.py
# uses, and deliberately specific for the same reason: a bare "timeout" also
# matches the shell's own "timeout: failed to run command".
XRT_FAIL_RE = re.compile(
    r"ERT_CMD_STATE|did not complete|not COMPLETED|command timeout|xrt::error", re.I
)
# A length the build cannot express, as opposed to a length that is slow or
# broken. Both are failures of the point, but only these say "this model cannot
# be built at this length", which is what the published cell has to mean.
BUILD_LIMIT_RES = (
    (
        re.compile(r"not in registry (GEMM_\S+)"),
        "no_registry_shape",
    ),
    (
        re.compile(r"Too many simultaneously active buffer descriptors[^\n]*"),
        "bd_exhaustion",
    ),
)


def _classify(out, rc):
    """A failed point's status, most specific cause first."""
    for rx, status in BUILD_LIMIT_RES:
        if rx.search(out):
            return status
    if XRT_FAIL_RE.search(out):
        return "device_fail"
    if rc != 0:
        m = re.search(r"error: .{0,90}|Killed|Cannot allocate|out of memory", out)
        return f"fail: {m.group(0)}" if m else "fail: rc=%d" % rc
    return "no_number"


def run_point(args, length, logdir):
    """Build + time one padded prefill length. Returns (point, raw output)."""
    env = dict(os.environ)
    # Both halves of the length. The Makefiles that export SEQ from CTX will
    # override the first with the same value; the ones that do not are the
    # reason it is set here at all.
    env[args.seq_env] = str(length)
    env[args.bench_env] = str(length)
    for kv in args.env or ():
        k, _, v = kv.partition("=")
        env[k] = v

    cmd = ["make", "-f", str(args.makefile), "profile-prefill", f"CTX={length}"]
    if args.peano_dir:
        cmd.append(f"PEANO_INSTALL_DIR={args.peano_dir}")
    cmd += list(args.make_var or ())

    pt = {"prefill_len": length}
    timed_out = False
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, env=env
        )
        out = r.stdout + r.stderr
        rc = r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        rc, timed_out = -1, True

    # Write the log before any early return: a timeout is the failure mode most
    # in need of its output, and it used to be the one mode that discarded it.
    (logdir / f"prefill_L{length}.log").write_text(out)
    if timed_out:
        pt["status"] = "timeout"
        return pt, out

    # Guard the axis before reading any number off it: a driver that built its
    # engines at some other length still prints a perfectly parseable TTFT, and
    # that number belongs to a different point than the one on the x-axis.
    built = SEQ_RE.search(out)
    if built and int(built.group(1)) != length:
        pt["status"] = f"wrong_seq_len: engines built at {built.group(1)}"
        return pt, out

    ttft = TTFT_RE.search(out)
    if not ttft:
        pt["status"] = _classify(out, rc)
        return pt, out

    pt["ttft_ms"] = round(float(ttft.group(1)) * 1000.0, 2)
    pt["prefill_tokens_per_sec"] = round(length / float(ttft.group(1)), 1)
    b = BENCH_RE.search(out)
    if b:
        pt["npu_dispatch_ms"] = round(float(b.group(3)), 2)
        pt["host_ms"] = round(float(b.group(2)) - float(b.group(3)), 2)
    # The driver's own first-token gate, when it ran one at this length.
    if PARIS_RE.search(out):
        pt["first_token_gate"] = "pass"
    elif MISS_RE.search(out):
        pt["first_token_gate"] = "fail"
    pt["status"] = "ok"
    return pt, out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", required=True, help="example dir name")
    p.add_argument("--makefile", required=True, type=Path)
    p.add_argument(
        "--lengths",
        required=True,
        help="comma-separated padded prefill lengths, e.g. 512,1024,2048",
    )
    p.add_argument(
        "--expect-fail",
        default="",
        help="lengths allowed to fail without failing the run (see sweep_decode.py)",
    )
    p.add_argument("--seq-env", default="Q4NX_SEQ_LEN", help="engine seq_len env var")
    p.add_argument("--bench-env", default="Q4NX_BENCH_L", help="bench length env var")
    p.add_argument("--peano-dir", default="")
    p.add_argument("--env", action="append", help="extra KEY=VALUE for the build/run")
    p.add_argument("--make-var", action="append", help="extra VAR=VALUE for make")
    p.add_argument("--timeout", type=int, default=3600, help="per-point seconds")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    expect_fail = {int(x) for x in args.expect_fail.split(",") if x.strip()}
    logdir = args.out.parent
    logdir.mkdir(parents=True, exist_ok=True)

    points, hard_fail = [], False
    for length in lengths:
        print(f"[sweep_prefill] {args.model_name} L={length} ...", flush=True)
        pt, _ = run_point(args, length, logdir)
        if pt["status"] != "ok":
            if length in expect_fail:
                # Mirror sweep_decode.py exactly: the published status becomes
                # "expected_fail" and the real cause moves to `detail`. A
                # separate boolean would not reach the dashboard -- both
                # append_history.py and the renderer key off `status`, so an
                # expected failure would have rendered as an unexpected one.
                pt["detail"] = pt["status"]
                pt["status"] = "expected_fail"
            else:
                hard_fail = True
        print(f"[sweep_prefill]   {json.dumps(pt)}", flush=True)
        points.append(pt)

    rec = {
        "model": args.model_name,
        "axis": "prefill_len",
        "metric": "ttft_ms",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "points": points,
    }
    args.out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    # An unexpected failure reds the sweep; an --expect-fail one is recorded and
    # published with its status, exactly as the decode sweep does.
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
