# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Sweep decode throughput against context length for a RUNTIME-context design.

The second decode sweeper, and the split is the mechanism, not the metric: it
writes the same record `sweep_decode.py` does, so the published table cannot
tell the two apart.

`sweep_decode.py` drives the fused-decode family, whose context is COMPILED IN
(ATTN_MAXL): every point is a template build, so it shells make once per
context, guards that a new xclbin actually appeared, and dispatches a separate
bench binary. None of that applies here. The bf16 models run their decode
attention on the host and their NPU kernels are per-token GEMVs with no context
dependence, so the context is a runtime value: one build, one process, one
resident set of weight BOs, and the whole list walked inside it. Trying to
express that as a mode of the other file would have meant a script where most
flags are inert for half its callers.

So this is the thin `sweep_prefill.py`-shaped runner: shell
`make profile-decode CONTEXTS=...` ONCE and parse what it prints. Everything
expensive (weight load, BO preload) is paid once for the whole curve rather
than once per point.

Latency only, never a correctness gate -- `make verify` is that. The contents
of the KV cache are synthetic; a context is set by sizing the cache and telling
the step which position to attend from, not by prefilling a real prompt.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# "[bench] decode ctx=8192 mean 380.625 ms (2.63 tok/s)" -- the line the bf16
# drivers print per context.
POINT_RE = re.compile(
    r"^\[bench\] decode ctx=(\d+)\s+mean\s+([\d.]+)\s*ms\s*\(([\d.]+)\s*tok/s\)", re.M
)

# Device-side "the dispatch did not complete" evidence. Same list sweep_decode.py
# uses, and deliberately specific for the same reason: a bare "timeout" also
# matches the shell's own "timeout: failed to run command".
XRT_FAIL_RE = re.compile(
    r"ERT_CMD_STATE|did not complete|not COMPLETED|command timeout|xrt::error", re.I
)
# A context the HOST could not hold. Distinct from a device failure because the
# published cell has to mean "this box ran out of RAM for the KV cache", which
# is a property of the runner, not of the design -- the same reason
# sweep_decode.py records 64k/128k separately.
OOM_RE = re.compile(r"std::bad_alloc|MemoryError|Cannot allocate|Killed", re.I)


def _classify(out, rc):
    """A missing point's status, most specific cause first."""
    if OOM_RE.search(out):
        return "host_oom"
    if XRT_FAIL_RE.search(out):
        return "xrt_incomplete"
    if rc != 0:
        m = re.search(r"error: .{0,90}|Traceback", out)
        return f"fail: {m.group(0)}" if m else f"fail: rc={rc}"
    return "no_number"


# Statuses --expect-fail is allowed to forgive. Same rule sweep_decode.py
# applies: only a point that reached the hardware and could not be held or
# completed. A build or import failure at an expected-fail context is still
# hard, or a broken build would be indistinguishable from a known runner limit.
EXPECTABLE = frozenset({"xrt_incomplete", "host_oom", "timeout"})


def run_sweep(args, contexts, logdir):
    """One `make profile-decode` over the whole list. Returns (points, output)."""
    env = dict(os.environ)
    for kv in args.env or ():
        k, _, v = kv.partition("=")
        env[k] = v

    cmd = ["make", "-f", str(args.makefile), "profile-decode"]
    cmd.append("CONTEXTS=" + ",".join(str(c) for c in contexts))
    if args.peano_dir:
        cmd.append(f"PEANO_INSTALL_DIR={args.peano_dir}")
    cmd += list(args.make_var or ())

    timed_out = False
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, env=env
        )
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        rc, timed_out = -1, True

    (logdir / "decode_sweep.log").write_text(out)

    # Parse what DID get printed. A run that dies partway still measured every
    # context before the one that killed it, and those points are real -- the
    # single-process design is what makes losing them otherwise likely, so they
    # are kept rather than discarding the whole curve.
    got = {
        int(m.group(1)): (float(m.group(2)), float(m.group(3)))
        for m in POINT_RE.finditer(out)
    }
    missing = _classify(out, rc) if not timed_out else "timeout"

    points = []
    for ctx in contexts:
        rec = {
            "context_len": ctx,
            "decode_tokens_per_sec": None,
            "ms_per_token": None,
            "status": "ok",
        }
        if ctx in got:
            rec["ms_per_token"], rec["decode_tokens_per_sec"] = got[ctx]
        else:
            rec["status"] = missing
        points.append(rec)
    return points, out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", required=True, help="example dir name")
    p.add_argument("--makefile", required=True, type=Path)
    p.add_argument(
        "--contexts", required=True, help="comma-separated KV depths, e.g. 1024,2048"
    )
    p.add_argument(
        "--expect-fail",
        default="",
        help="contexts allowed to fail without failing the run, and ONLY with a "
        f"host/device status ({'/'.join(sorted(EXPECTABLE))})",
    )
    p.add_argument("--peano-dir", default="")
    p.add_argument("--env", action="append", help="extra KEY=VALUE for the run")
    p.add_argument("--make-var", action="append", help="extra VAR=VALUE for make")
    p.add_argument("--timeout", type=int, default=5400, help="whole-sweep seconds")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    expect_fail = {int(c) for c in args.expect_fail.split(",") if c.strip()}
    logdir = args.out.parent
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"[sweep_decode_runtime] {args.model_name} {contexts} ...", flush=True)
    points, _ = run_sweep(args, contexts, logdir)

    hard_fail = False
    for pt in points:
        if pt["status"] != "ok":
            if pt["context_len"] in expect_fail and pt["status"] in EXPECTABLE:
                # Mirror sweep_decode.py exactly: the published status becomes
                # "expected_fail" and the real cause moves to `detail`. Both
                # append_history.py and the renderer key off `status`.
                pt["detail"] = pt["status"]
                pt["status"] = "expected_fail"
            else:
                hard_fail = True
        tps = pt["decode_tokens_per_sec"]
        print(
            f"[sweep] {args.model_name} ctx={pt['context_len']:<7} "
            f"{f'{tps:.2f} tok/s' if tps is not None else pt['status']}",
            flush=True,
        )

    args.out.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "axis": "context_len",
                "metric": "decode_tokens_per_sec",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "runner": os.environ.get("RUNNER_NAME", ""),
                "points": points,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[sweep] wrote {args.out}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
