# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Paired Full20 or Full80 study, with separate processes and matched inputs."""

import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main(*, model, model_dir, verify_inputs):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-dataset", type=int, choices=(4, 16), default=4)
    a = p.parse_args()
    cohort = verify_inputs(a.inputs)
    counts = Counter()
    cases = []
    for case in cohort["cases"]:
        if counts[case["dataset"]] < a.per_dataset:
            cases.append(case)
            counts[case["dataset"]] += 1
    expected_counts = {name: a.per_dataset for name in cohort["dataset_order"]}
    assert counts == expected_counts
    expected_ids = {c["id"]: (c["cache_sha256"], len(c["input_ids"])) for c in cases}
    requests = 5 * a.per_dataset
    assert len(expected_ids) == requests
    a.output.parent.mkdir(parents=True, exist_ok=True)
    work = a.output.parent / (a.output.stem + "-requests")
    work.mkdir(exist_ok=False)
    results, ids, elapsed = {}, {}, {}
    for engine in ("b1", "dflash"):
        raw = work / (engine + ".jsonl")
        cmd = [
            sys.executable,
            str(model_dir / "benchmark.py"),
            "--engine",
            engine,
            "--inputs",
            str(a.inputs),
            "--output",
            str(raw),
            "--per-dataset",
            str(a.per_dataset),
            "--target",
            str(a.build / ("baseline" if engine == "b1" else "target")),
        ]
        if engine == "dflash":
            cmd += [
                "--draft",
                str(a.build / "draft"),
                "--prepass",
                str(a.build / "prepass"),
            ]
        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        elapsed[engine] = time.perf_counter() - start
        summary = json.loads(raw.with_suffix(".summary.json").read_text())
        rows = [json.loads(s) for s in raw.read_text().splitlines()]
        assert (
            summary["status"] == "completed"
            and summary["completed"] == len(rows) == requests
        )
        assert {
            k: v["requests"] for k, v in summary["per_dataset"].items()
        } == expected_counts
        assert Counter(r["dataset"] for r in rows) == expected_counts
        ids[engine] = {r["id"]: (r["prefill_sha256"], r["input_tokens"]) for r in rows}
        assert ids[engine] == expected_ids, "Engine did not measure the fixed cohort"
        assert [r["id"] for r in rows] == [c["id"] for c in cases]
        results[engine] = summary
    assert ids["b1"] == ids["dflash"], "Engines measured different inputs"

    def compare(base, spec):
        return dict(
            b1_tokens_per_second=base["tokens_per_second"],
            dflash_tokens_per_second=spec["tokens_per_second"],
            speedup=spec["tokens_per_second"] / base["tokens_per_second"],
            emitted_per_round=spec["emitted_per_round"],
            acceptance_rate=spec["acceptance_rate"],
        )

    result = dict(
        model=model,
        status="completed",
        requests_per_engine=requests,
        samples_per_dataset=a.per_dataset,
        protocol=f"dflash-full{requests}-saved-prefill-v1",
        selection="First N cases per dataset in the frozen seed-0 manifest order",
        selected_ids=[c["id"] for c in cases],
        passes=1,
        max_new_tokens=2048,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        overall=compare(results["b1"]["overall"], results["dflash"]["overall"]),
        per_dataset={
            k: compare(
                results["b1"]["per_dataset"][k], results["dflash"]["per_dataset"][k]
            )
            for k in results["b1"]["per_dataset"]
        },
        process_seconds=elapsed,
        records={k: str(work / (k + ".jsonl")) for k in results},
        engines=results,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall"], indent=2))
