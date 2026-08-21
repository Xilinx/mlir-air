# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Round-trip checks for the published LLM benchmark tables.

The nightly's numbers reach the docs site through three hops:

    perf.json / sweep.json  ->  append_history.py  ->  *history.ndjson
                            ->  generate_readme.py ->  docs/llms/index.md

Every published defect so far has been a field or a distinction lost at one of
those hops while both ends kept working: an empty perf.json blanking the whole
section, verify_status not surviving the sweep round-trip so the Verify column
went empty, runner vanishing from the provenance line, and a build failure
rendering as the same dash as a context the box could not reach. None of them
fail loudly -- the page just quietly says less than the run measured.

So these assert on the seams rather than on the numbers. No hardware, no build,
no network:

    python3 programming_examples/test_perf_publish.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APPEND = HERE / "llms" / "bench" / "append_history.py"

sys.path.insert(0, str(HERE))
from generate_readme import (  # noqa: E402
    load_llm_history,
    load_llm_sweep_history,
    render_llm_benchmark,
    render_llm_sweep,
)

TOOLCHAIN = {
    "mlir_air_sha": "d97d4e8c8eaeb54cb04289feb349221ed4a78bac",
    "mlir_aie_hash": "7e00b57955e108fe9d8e9419f5828a0c7e650858",
    "llvm_aie_version": "21.0.0.2026080601+f4a72c27",
}
RUNNER = "amdryzenai5pro340"

PERF = [
    {
        "model": "toy_1b",
        "timestamp_utc": "2026-08-21T10:31:45Z",
        "runner": RUNNER,
        "verify_status": "pass",
        "metrics": {
            "ttft_ms": 900.0,
            "decode_tokens_per_sec": 42.0,
            "context_len": 2048,
        },
        "toolchain": TOOLCHAIN,
    }
]

SWEEP = [
    {
        "model": "toy_1b_q4nx",
        "timestamp_utc": "2026-08-21T10:31:45Z",
        "runner": RUNNER,
        "verify_status": "pass",
        "toolchain": TOOLCHAIN,
        "points": [
            {
                "context_len": 1024,
                "decode_tokens_per_sec": 60.0,
                "ms_per_token": 16.6,
                "status": "ok",
            },
            {
                "context_len": 65536,
                "decode_tokens_per_sec": None,
                "ms_per_token": None,
                "status": "expected_fail",
            },
        ],
    },
    {
        "model": "toy_3b_q4nx",
        "timestamp_utc": "2026-08-21T10:31:45Z",
        "runner": RUNNER,
        "verify_status": "skip",
        "toolchain": TOOLCHAIN,
        "points": [
            {
                "context_len": 1024,
                "decode_tokens_per_sec": None,
                "ms_per_token": None,
                "status": "build_fail",
            },
            {
                "context_len": 65536,
                "decode_tokens_per_sec": None,
                "ms_per_token": None,
                "status": "expected_fail",
            },
        ],
    },
]

FAILURES = []


def check(cond, what):
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        FAILURES.append(what)


def append(src_flag, payload, history, run_id):
    """Run append_history.py the way nightlyPerfBenchmark.yml does."""
    src = history.parent / f"{src_flag}.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    r = subprocess.run(
        [
            sys.executable,
            str(APPEND),
            f"--{src_flag}",
            str(src),
            "--history",
            str(history),
            "--run-id",
            str(run_id),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        hist, shist = tmp / "history.ndjson", tmp / "sweep_history.ndjson"

        print("append_history.py -> ndjson")
        append("perf", PERF, hist, 1)
        append("sweep", SWEEP, shist, 1)
        prow = json.loads(hist.read_text(encoding="utf-8").splitlines()[0])
        srow = json.loads(shist.read_text(encoding="utf-8").splitlines()[0])
        # The two fields #1850 dropped. Asserted on the raw row, because the
        # renderer defaults them to "" and a missing field looks identical to
        # an empty one by the time it reaches the page.
        check(prow.get("verify_status") == "pass", "perf row keeps verify_status")
        check(prow.get("runner") == RUNNER, "perf row keeps runner")
        check(srow.get("verify_status") == "pass", "sweep row keeps verify_status")
        check(srow.get("runner") == RUNNER, "sweep row keeps runner")
        check(srow.get("status") == "ok", "sweep row keeps the point's own status")

        check("no change" in append("sweep", SWEEP, shist, 1), "re-append is a no-op")

        print("ndjson -> loaders")
        recs = load_llm_history(hist)
        curves = load_llm_sweep_history(shist)
        check(recs[0]["verify_status"] == "pass", "history load keeps verify_status")
        check(recs[0]["runner"] == RUNNER, "history load keeps runner")
        byname = {c["model"]: c for c in curves}
        check(
            byname["toy_1b_q4nx"].get("verify_status") == "pass"
            and byname["toy_3b_q4nx"].get("verify_status") == "skip",
            "sweep load carries verify_status onto each curve",
        )

        print("loaders -> rendered page")
        page = render_llm_benchmark(
            None, sweep_recs=curves, history_path=hist  # no perf.json at all
        )
        check(bool(page), "an absent perf.json does not blank the section")
        check(f"runner {RUNNER}" in page, "provenance names the runner")
        check("| 🟢 |" in page, "scalar table renders a verify badge")

        sweep_md = render_llm_sweep(curves)
        check(
            sweep_md.count("🟢") == 1 and sweep_md.count("⚪") == 1,
            "sweep table renders one badge per model",
        )
        check("| 60.00 |" in sweep_md, "a measured point renders its number")
        check("✗" in sweep_md, "a build_fail renders as a failure, not a dash")
        check("—" in sweep_md, "an expected_fail still renders as a dash")
        check(
            "not a limit of the machine" in sweep_md
            and "the limit is the machine's" in sweep_md,
            "both markers present -> both legends shown",
        )

        # Legend must not explain a marker no cell uses.
        clean = render_llm_sweep(
            [
                {
                    "model": "m",
                    "verify_status": "pass",
                    "points": [
                        {
                            "context_len": 1024,
                            "decode_tokens_per_sec": 1.0,
                            "status": "ok",
                        }
                    ],
                }
            ]
        )
        check(
            "✗" not in clean and "—" not in clean,
            "an all-measured table carries no legend",
        )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
