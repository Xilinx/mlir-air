# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Append a nightly perf.json into the append-only history.ndjson time series.

The nightly LLM benchmark (nightlyPerfBenchmark.yml) produces a per-run
`perf.json` (a list of per-model records). This flattens each record into one
line of `history.ndjson` on the durable `perf-history` branch so the docs build
can plot TTFT / decode throughput over time.

Idempotent on run_id: if any row for this run_id already exists, nothing is
appended, so re-running the *same* workflow run does not double-count. (A newly
dispatched run gets a fresh run_id and is treated as new data, as intended.)

Usage:
  python3 append_history.py --perf perf.json --history history.ndjson --run-id 123
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
            "ttft_ms": m.get("ttft_ms"),
            "decode_tokens_per_sec": m.get("decode_tokens_per_sec"),
            "context_len": m.get("context_len"),
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
    ap.add_argument("--perf", required=True, help="Path to the nightly perf.json")
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

    perf_path = Path(args.perf)
    if not perf_path.is_file():
        print(f"no perf.json at {perf_path}; nothing to append")
        return 0
    try:
        recs = json.loads(perf_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        # Best-effort: a missing/corrupt/partial perf.json should not crash the
        # CI step (which is itself continue-on-error), just skip this run.
        print(f"could not read {perf_path} ({e}); nothing to append")
        return 0
    if not recs:
        print("perf.json is empty; nothing to append")
        return 0

    if run_id in _existing_run_ids(args.history):
        print(f"run {run_id} already in history; no change")
        return 0

    rows = list(_flat_rows(recs, run_id))
    hist = Path(args.history)
    hist.parent.mkdir(parents=True, exist_ok=True)
    with hist.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"appended {len(rows)} rows for run {run_id} to {hist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
