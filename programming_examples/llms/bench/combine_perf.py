#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Combine one benchmark run's per-model results into the files the docs publish.

Reads <build-dir>/test/llms/**/*.{perf,sweep,prefill_sweep,full20,full80,
accuracy,tasks}.json and verify.xml; writes perf.json, sweep.json,
prefill_sweep.json, full20.json, full80.json, accuracy.json, tasks.json and
perf_out/ in the current directory.
Used by the nightly and by publish_offline.py, so both publish the same shape.
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--aie-hash", required=True)
ap.add_argument("--peano", required=True, help="llvm-aie version")
ap.add_argument("--air-sha", required=True)
ap.add_argument("--runner", required=True, help="machine the results were measured on")
ap.add_argument("--build-dir", default="build_assert")
ap.add_argument("--verify-xml", default="verify.xml")
args = ap.parse_args()
aie_hash, peano_ver, air_sha = args.aie_hash, args.peano, args.air_sha
os.makedirs("perf_out", exist_ok=True)


# Include nested examples and explicitly declared profile model IDs.
from model_ids import verification_status

vstatus = verification_status(args.verify_xml, Path(__file__).resolve().parents[1])

recs = []
for f in sorted(
    glob.glob(f"{args.build_dir}/test/llms/**/*.perf.json", recursive=True)
):
    # A timeout or cancel truncates one model's file mid-write. That
    # used to raise here and kill the step, so nothing published at
    # all. Skip the file instead; the vstatus loop below gives the
    # model a null row if verify ran for it.
    try:
        d = json.load(open(f))
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: {f} unreadable ({e}); recording as failed", file=sys.stderr)
        continue
    d["runner"] = args.runner
    d["toolchain"] = {
        "mlir_air_sha": air_sha,
        "mlir_aie_hash": aie_hash,
        "llvm_aie_version": peano_ver,
    }
    d["verify_status"] = vstatus.get(d["model"], "")
    recs.append(d)
    prof = os.path.join(os.path.dirname(f), "profile.txt")
    if os.path.exists(prof):
        shutil.copy(prof, f"perf_out/{d['model']}.profile.txt")

# Surface models that ran verify but produced no profile record (their
# profile compile/run failed): emit a null-metric row so the failure is
# visible on the dashboard instead of the model silently disappearing.
# Skip "skip" models (e.g. gated qwen3_4b) — those are intentionally
# not benchmarked, not failures.
profiled = {d["model"] for d in recs}
now = (
    datetime.datetime.now(datetime.timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z")
)
for model, st in vstatus.items():
    if model in profiled or st == "skip":
        continue
    recs.append(
        {
            "model": model,
            "timestamp_utc": now,
            "runner": args.runner,
            "verify_status": st,
            "metrics": {
                "ttft_ms": None,
                "decode_tokens_per_sec": None,
                "context_len": None,
            },
            "toolchain": {
                "mlir_air_sha": air_sha,
                "mlir_aie_hash": aie_hash,
                "llvm_aie_version": peano_ver,
            },
            "run_params": {"n_tokens": None, "prompt": None},
        }
    )

recs.sort(key=lambda r: r["model"])
json.dump(recs, open("perf.json", "w"), indent=2)
print(json.dumps(recs, indent=2))

# Keep the two cohorts distinct, including their partial request logs.
for size in (20, 80):
    kind = f"full{size}"
    for request_dir in sorted(
        glob.glob(f"{args.build_dir}/test/llms/**/*.{kind}-requests", recursive=True)
    ):
        shutil.copytree(
            request_dir, f"perf_out/{os.path.basename(request_dir)}", dirs_exist_ok=True
        )
    studies = []
    for f in sorted(
        glob.glob(f"{args.build_dir}/test/llms/**/*.{kind}.json", recursive=True)
    ):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: unreadable {kind} result {f}: {e}", file=sys.stderr)
            continue
        if d.get("status") != "completed" or d.get("requests_per_engine") != size:
            print(f"WARNING: incomplete {kind} result {f}", file=sys.stderr)
            continue
        d["runner"] = args.runner
        d["toolchain"] = dict(
            mlir_air_sha=air_sha, mlir_aie_hash=aie_hash, llvm_aie_version=peano_ver
        )
        d["verify_status"] = vstatus.get(d["model"], "")
        # Prompt lengths live only in the per-request log; the docs
        # page states the range the study covered.
        log = os.path.join(
            os.path.dirname(f), (d.get("records") or {}).get("dflash", "")
        )
        try:
            lens = [json.loads(l)["input_tokens"] for l in open(log) if l.strip()]
            d["prompt_tokens"] = {"min": min(lens), "max": max(lens)}
        except (OSError, ValueError, KeyError) as e:
            print(f"WARNING: no prompt lengths from {log}: {e}", file=sys.stderr)
        studies.append(d)
    json.dump(studies, open(f"{kind}.json", "w"), indent=2)

# Decode tok/s-vs-context sweeps, kept in their own file rather than
# folded into perf.json: a sweep is a curve, not a scalar, and
# history.ndjson's one-row-per-model shape cannot hold one. The
# models that have a sweep are published from this file instead of the
# perf.json table (generate_readme.py drops them from the latter).
sweeps = []
for f in sorted(
    glob.glob(f"{args.build_dir}/test/llms/**/*.sweep.json", recursive=True)
):
    d = json.load(open(f))
    d["runner"] = args.runner
    d["toolchain"] = {
        "mlir_air_sha": air_sha,
        "mlir_aie_hash": aie_hash,
        "llvm_aie_version": peano_ver,
    }
    d["verify_status"] = vstatus.get(d["model"], "")
    sweeps.append(d)
sweeps.sort(key=lambda r: r["model"])
json.dump(sweeps, open("sweep.json", "w"), indent=2)
print(
    f"{len(sweeps)} sweep(s):",
    ", ".join(f"{s['model']}({len(s['points'])} pts)" for s in sweeps),
)

# Prefill TTFT curves, in a third file for the same reason sweep.json is
# a second one: a different x-axis (padded prefill length) and a
# different metric (ttft_ms), which neither of the other two rows can
# hold without changing what an existing row means.
pf = []
for f in sorted(
    glob.glob(f"{args.build_dir}/test/llms/**/*.prefill_sweep.json", recursive=True)
):
    try:
        d = json.load(open(f))
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: {f} unreadable ({e}); skipping", file=sys.stderr)
        continue
    d["runner"] = args.runner
    d["toolchain"] = {
        "mlir_air_sha": air_sha,
        "mlir_aie_hash": aie_hash,
        "llvm_aie_version": peano_ver,
    }
    d["verify_status"] = vstatus.get(d["model"], "")
    pf.append(d)
pf.sort(key=lambda r: r["model"])
json.dump(pf, open("prefill_sweep.json", "w"), indent=2)
print(
    f"{len(pf)} prefill sweep(s):",
    ", ".join(f"{s['model']}({len(s['points'])} pts)" for s in pf),
)

# Accuracy records (verify/accuracy.py), one per (model, reference, path,
# context), go to their own file: they compare against a reference, which no
# perf record does.
acc = []
for f in sorted(
    glob.glob(f"{args.build_dir}/test/llms/**/*.accuracy.json", recursive=True)
):
    try:
        d = json.load(open(f))
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: {f} unreadable ({e}); skipping", file=sys.stderr)
        continue
    d["runner"] = args.runner
    d["toolchain"] = {
        "mlir_air_sha": air_sha,
        "mlir_aie_hash": aie_hash,
        "llvm_aie_version": peano_ver,
    }
    acc.append(d)
acc.sort(
    key=lambda r: (r["model"], r["reference"]["kind"], r["path"], r.get("ctx") or 0)
)
json.dump(acc, open("accuracy.json", "w"), indent=2)
print(f"{len(acc)} accuracy record(s)")

# Task score records (verify/lm_eval_npu.py), one per (model, lm_eval run).
tasks = []
for f in sorted(
    glob.glob(f"{args.build_dir}/test/llms/**/*.tasks.json", recursive=True)
):
    try:
        d = json.load(open(f))
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: {f} unreadable ({e}); skipping", file=sys.stderr)
        continue
    d["runner"] = args.runner
    d["toolchain"] = {
        "mlir_air_sha": air_sha,
        "mlir_aie_hash": aie_hash,
        "llvm_aie_version": peano_ver,
    }
    tasks.append(d)
tasks.sort(key=lambda r: (r["model"], sorted(r.get("tasks") or {})))
json.dump(tasks, open("tasks.json", "w"), indent=2)
print(f"{len(tasks)} task score record(s)")
