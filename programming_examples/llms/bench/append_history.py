# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Append a nightly perf.json / sweep.json into an append-only ndjson series.

The nightly LLM benchmark (nightlyPerfBenchmark.yml) produces a per-run
`perf.json` (a list of per-model records). This flattens each record into one
line of `history.ndjson` on the durable `perf-history` branch so the docs build
can plot TTFT / decode throughput over time.

Idempotent on run_id: if any row for this run_id already exists, nothing is
appended, so re-running the *same* workflow run does not double-count. (A newly
dispatched run gets a fresh run_id and is treated as new data, as intended.)

The two series are kept in separate files. history.ndjson is one row per model
per run and every row in it was taken at the profile prompt's ~6 tokens of
context; sweep_history.ndjson is one row per (model, context) per run. Folding
the sweep into the first would silently change what the plotted number means
partway through the series, which reads as a step change that never happened.

Usage:
  python3 append_history.py --perf  perf.json  --history history.ndjson       --run-id 123
  python3 append_history.py --sweep sweep.json --history sweep_history.ndjson --run-id 123
"""

import argparse
import json
import sys
from pathlib import Path


def _flat_rows(recs, run_id):
    """Yield one flat history row per perf.json record."""
    for d in recs:
        m = d.get("metrics", {}) or {}
        tc = d.get("toolchain", {}) or {}
        ts = d.get("timestamp_utc", "") or ""
        yield {
            "date": ts[:10],
            "timestamp_utc": ts,
            "run_id": run_id,
            "air_sha": tc.get("mlir_air_sha", ""),
            "aie_hash": tc.get("mlir_aie_hash", ""),
            "peano": tc.get("llvm_aie_version", ""),
            "model": d.get("model", ""),
            "runner": d.get("runner", ""),
            "ttft_ms": m.get("ttft_ms"),
            "decode_tokens_per_sec": m.get("decode_tokens_per_sec"),
            "context_len": m.get("context_len"),
            "verify_status": d.get("verify_status", ""),
        }


def _flat_sweep_rows(recs, run_id):
    """Yield one flat history row per (model, context) sweep point.

    Points that did not produce a number are still emitted, with a null metric
    and their status, so a context that starts failing is visible in the series
    rather than the curve just getting shorter.

    `status` is the point's own outcome (did this context produce a number);
    `verify_status` is the model's `make verify` result for the whole run, which
    is what the published Verify column reports. They are different things, and
    both are per-row here because the docs build reads only this file.
    """
    for d in recs:
        tc = d.get("toolchain", {}) or {}
        ts = d.get("timestamp_utc", "") or ""
        for pt in d.get("points", []) or []:
            yield {
                "date": ts[:10],
                "timestamp_utc": ts,
                "run_id": run_id,
                "air_sha": tc.get("mlir_air_sha", ""),
                "aie_hash": tc.get("mlir_aie_hash", ""),
                "peano": tc.get("llvm_aie_version", ""),
                "model": d.get("model", ""),
                "runner": d.get("runner", ""),
                "context_len": pt.get("context_len"),
                "decode_tokens_per_sec": pt.get("decode_tokens_per_sec"),
                "ms_per_token": pt.get("ms_per_token"),
                "status": pt.get("status", ""),
                "verify_status": d.get("verify_status", ""),
            }


def _existing_run_ids(history_path):
    """Return the set of run_ids already recorded in history.ndjson.

    Streamed line-by-line so memory stays bounded as the history grows.
    """
    ids = set()
    p = Path(history_path)
    if not p.is_file():
        return ids
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line).get("run_id"))
            except json.JSONDecodeError:
                continue
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--perf", help="Path to the nightly perf.json")
    src.add_argument("--sweep", help="Path to the nightly sweep.json")
    ap.add_argument(
        "--history", required=True, help="Path to history.ndjson (created if absent)"
    )
    ap.add_argument(
        "--run-id",
        required=True,
        help="Unique id for this nightly run (dedup key, e.g. github.run_id)",
    )
    args = ap.parse_args()

    try:
        run_id = int(args.run_id)
    except ValueError:
        run_id = args.run_id

    src_path = Path(args.perf or args.sweep)
    flatten = _flat_rows if args.perf else _flat_sweep_rows
    if not src_path.is_file():
        print(f"no {src_path.name}; nothing to append")
        return 0
    try:
        recs = json.loads(src_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        # Best-effort: a missing/corrupt/partial input should not crash the CI
        # step (which is itself continue-on-error), just skip this run.
        print(f"could not read {src_path} ({e}); nothing to append")
        return 0
    if not recs:
        print(f"{src_path.name} is empty; nothing to append")
        return 0

    if run_id in _existing_run_ids(args.history):
        print(f"run {run_id} already in history; no change")
        return 0

    rows = list(flatten(recs, run_id))
    hist = Path(args.history)
    hist.parent.mkdir(parents=True, exist_ok=True)
    with hist.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"appended {len(rows)} rows for run {run_id} to {hist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
