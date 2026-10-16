# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""One Full80 pass per engine, with separate processes and matched input/timing."""

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    work = a.output.parent / (a.output.stem + "-requests")
    work.mkdir(exist_ok=False)
    results, ids, elapsed = {}, {}, {}
    for engine in ("b1", "dflash"):
        raw = work / (engine + ".jsonl")
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("benchmark.py")),
            "--engine",
            engine,
            "--inputs",
            str(a.inputs),
            "--output",
            str(raw),
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
            summary["status"] == "completed" and summary["completed"] == len(rows) == 80
        )
        assert len(summary["per_dataset"]) == 5
        assert all(x["requests"] == 16 for x in summary["per_dataset"].values())
        ids[engine] = {r["id"]: (r["prefill_sha256"], r["input_tokens"]) for r in rows}
        assert len(ids[engine]) == 80
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
        model="qwen3_4b_dflash_b8",
        status="completed",
        requests_per_engine=80,
        protocol="dflash-full80-saved-prefill-v1",
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


if __name__ == "__main__":
    main()
