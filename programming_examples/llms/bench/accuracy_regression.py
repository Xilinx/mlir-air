# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Flag accuracy regressions in accuracy_history.ndjson.

A series is one (model, reference kind, NPU path, context, scored window). Its
newest row is compared with the median of the rows before it, at most WINDOW of
them and at least MIN_PRIOR, and flagged when

  * same top-1 falls by more than SIGMAS combined standard errors, or
  * mean KLD at least doubles and the rise is more than SIGMAS combined
    standard errors.

Both tests need the change to be large against the measurement's own noise,
and the KLD one also large in relative terms, so run-to-run jitter in a
low-KLD series does not trip it.

Usage:
  python3 accuracy_regression.py accuracy_history.ndjson [--fail]
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

WINDOW = 4
MIN_PRIOR = 2
SIGMAS = 3.0
KLD_FACTOR = 2.0


def series_key(row):
    return (
        row.get("model"),
        row.get("ref_kind"),
        row.get("path") or "decode",
        row.get("ctx"),
        row.get("scored_from"),
    )


def load(path):
    rows = []
    p = Path(path)
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _combined_se(a, b):
    return math.sqrt((a or 0.0) ** 2 + (b or 0.0) ** 2)


def regressions(rows, window=WINDOW, min_prior=MIN_PRIOR, sigmas=SIGMAS):
    """One dict per flagged series: its key, the metric, new and baseline values."""
    by_series = {}
    for r in rows:
        if r.get("kld_mean") is None or r.get("same_top") is None:
            continue
        by_series.setdefault(series_key(r), []).append(r)
    found = []
    for key, rs in sorted(by_series.items(), key=lambda kv: str(kv[0])):
        rs.sort(key=lambda r: r.get("timestamp_utc") or "")
        new, prior = rs[-1], rs[-1 - window : -1]
        if len(prior) < min_prior:
            continue
        top = statistics.median(r["same_top"] for r in prior)
        top_se = statistics.median(r.get("same_top_se") or 0.0 for r in prior)
        if top - new["same_top"] > sigmas * _combined_se(
            new.get("same_top_se"), top_se
        ):
            found.append(_flag(key, new, "same_top", new["same_top"], top))
        kld = statistics.median(r["kld_mean"] for r in prior)
        kld_se = statistics.median(r.get("kld_se") or 0.0 for r in prior)
        rise = new["kld_mean"] - kld
        if new["kld_mean"] >= KLD_FACTOR * kld and rise > sigmas * _combined_se(
            new.get("kld_se"), kld_se
        ):
            found.append(_flag(key, new, "kld_mean", new["kld_mean"], kld))
    return found


def _flag(key, row, metric, value, baseline):
    model, ref_kind, path, ctx, scored_from = key
    return {
        "model": model,
        "ref_kind": ref_kind,
        "path": path,
        "ctx": ctx,
        "scored_from": scored_from,
        "metric": metric,
        "value": value,
        "baseline": baseline,
        "timestamp_utc": row.get("timestamp_utc"),
        "air_sha": row.get("air_sha"),
    }


def markdown(found):
    if not found:
        return "No accuracy regressions against the recent history.\n"
    lines = [
        "| model | reference | path | ctx | metric | now | recent median |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for f in found:
        lines.append(
            f"| {f['model']} | {f['ref_kind']} | {f['path']} | {f['ctx']} | "
            f"{f['metric']} | {f['value']:.4f} | {f['baseline']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("history")
    ap.add_argument("--fail", action="store_true", help="exit 1 on any regression")
    args = ap.parse_args()
    found = regressions(load(args.history))
    sys.stdout.write(markdown(found))
    return 1 if (found and args.fail) else 0


if __name__ == "__main__":
    sys.exit(main())
