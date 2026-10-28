#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Publish LLM results measured off the nightly runner to the docs page.

The nightly runner's disk holds the models up to 4B (hf_models.txt); the 7-8B
examples (hf_models_offline.txt) run on another Krackan machine. After running
their lit suites there, e.g. from the build directory:

  LIT_OPTS="--xunit-xml-output $PWD/../verify.xml --filter <models>" \\
    ninja check-programming-examples-llms-verify
  ninja check-programming-examples-llms-{profile,sweep,prefill-sweep,full20}

this combines the results the way the nightly does (combine_perf.py), appends
them to the `perf-history` branch marked source=offline, and asks the docs
workflow to rebuild. Only models the nightly does not run are published, so an
offline row never displaces the runner's own number.

Accuracy (verify/accuracy.py) runs only off the runner, for every model:

  ninja check-programming-examples-llms-accuracy

and its records are published for all models. Left without --verify-xml, a
run publishes accuracy only.
"""

import argparse
import datetime
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# Output file -> its series on the perf-history branch, the flag
# append_history.py reads it with, and whether only models the nightly skips
# are published (the series the nightly also measures).
SERIES = [
    ("perf.json", "history.ndjson", "--perf", True),
    ("sweep.json", "sweep_history.ndjson", "--sweep", True),
    ("prefill_sweep.json", "prefill_sweep_history.ndjson", "--prefill-sweep", True),
    ("full20.json", "spec_decode_history.ndjson", "--spec-decode", True),
    ("accuracy.json", "accuracy_history.ndjson", "--accuracy", False),
]
LLMS = REPO / "programming_examples/llms"
SUITES = re.compile(r"run_npu2_(verify|profile|sweep|prefill_sweep|full20)")


def run(*cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def output(*cmd, cwd=None):
    return subprocess.run(
        [str(c) for c in cmd], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def aie_hash():
    """The pinned mlir-aie hash, read the way the nightly reads it."""
    for line in (REPO / "utils/clone-mlir-aie.sh").read_text().splitlines():
        key, sep, rest = line.partition("=")
        if sep and re.search(r"(^| )HASH$", key):
            return rest.split("=")[0]
    sys.exit("could not parse HASH from utils/clone-mlir-aie.sh")


def _hf_feature(repo_id):
    # Same normalization as lit.cfg.py's hfweights_* features.
    return "hfweights_" + re.sub(r"[^a-z0-9]+", "_", repo_id.lower()).strip("_")


def offline_models():
    """Model ids of the examples the nightly skips because hf_models.txt does not
    seed their weights: lit's own gate, evaluated against the runner's list."""
    seeded = set()
    for line in (LLMS / "hf_models.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            seeded.add(_hf_feature(line.partition("@")[0]))
    nightly, ids = set(), {}
    for lit in LLMS.rglob("run_npu2_*.lit"):
        if not SUITES.match(lit.name):
            continue
        directory = lit.parent.relative_to(LLMS).as_posix()
        text = lit.read_text()
        if set(re.findall(r"hfweights_[a-z0-9_]+", text)) <= seeded:
            nightly.add(directory)
        names = ids.setdefault(directory, {directory.replace("/", "_")})
        names.update(re.findall(r"^//\s*PERF_MODEL:\s*(\S+)", text, re.M))
        names.update(re.findall(r"--model(?:-name)?\s+(\S+)", text))
    return set().union(*(v for d, v in ids.items() if d not in nightly))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--machine", required=True, help="name of the machine measured on")
    ap.add_argument("--build-dir", required=True, help="the build tree lit ran in")
    ap.add_argument(
        "--verify-xml",
        default=None,
        help="lit's xunit output of verify; without it only accuracy is published",
    )
    ap.add_argument("--air-sha", default=None, help="default: this checkout's HEAD")
    ap.add_argument("--peano", default=None, help="default: installed llvm-aie")
    ap.add_argument("--remote", default="origin")
    ap.add_argument(
        "--dry-run", action="store_true", help="append locally; do not push or rebuild"
    )
    args = ap.parse_args()

    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    run_id = f"offline-{args.machine}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    air_sha = args.air_sha or output("git", "rev-parse", "HEAD", cwd=REPO)
    peano = args.peano or importlib.metadata.version("llvm-aie")
    url = output("git", "remote", "get-url", args.remote, cwd=REPO)
    work = Path(tempfile.mkdtemp(prefix="publish_offline_"))
    print(f"run {run_id}, mlir-air {air_sha[:7]}, llvm-aie {peano}; work dir {work}")

    run(
        sys.executable,
        HERE / "combine_perf.py",
        "--aie-hash",
        aie_hash(),
        "--peano",
        peano,
        "--air-sha",
        air_sha,
        "--runner",
        args.machine,
        "--build-dir",
        Path(args.build_dir).resolve(),
        # Without one, combine_perf finds no verify results: none are published.
        "--verify-xml",
        Path(args.verify_xml).resolve() if args.verify_xml else work / "verify.xml",
        cwd=work,
    )

    offline = offline_models()
    # A push can lose to the nightly's; start again from a fresh clone, which
    # append_history.py makes safe (it skips a run_id already present).
    for attempt in range(1, 4):
        hist = work / f"perf-history-{attempt}"
        run(
            "git", "clone", "-q", "--branch", "perf-history", "--depth", "50", url, hist
        )
        published = set()
        for name, series, flag, offline_only in SERIES:
            # Without verify results a perf record would publish an empty
            # verify status over the model's last one.
            if offline_only and not args.verify_xml:
                continue
            recs = (
                json.loads((work / name).read_text()) if (work / name).is_file() else []
            )
            keep = [r for r in recs if not offline_only or r.get("model") in offline]
            published |= {r.get("model") for r in keep}
            dropped = sorted(
                {r.get("model") for r in recs} - {r.get("model") for r in keep}
            )
            if dropped:
                print(
                    f"{name}: not publishing {', '.join(dropped)} (nightly runs them)"
                )
            if not keep:
                continue
            filtered = work / f"offline_{name}"
            filtered.write_text(json.dumps(keep))
            run(
                sys.executable,
                HERE / "append_history.py",
                flag,
                filtered,
                "--history",
                hist / series,
                "--run-id",
                run_id,
                "--source",
                "offline",
            )
        if not published:
            print("nothing to publish")
            return 0
        print(f"publishing {', '.join(sorted(published))}")
        accuracy_history = hist / "accuracy_history.ndjson"
        if accuracy_history.is_file():
            from accuracy_regression import load, markdown, regressions

            print(markdown(regressions(load(accuracy_history))))
        run("git", "add", "-A", cwd=hist)
        run(
            "git",
            "commit",
            "-q",
            "-m",
            f"perf-history: offline {args.machine} {air_sha[:7]} run {run_id}",
            cwd=hist,
        )
        if args.dry_run:
            print(f"dry run: not pushed; inspect {hist}")
            return 0
        if (
            subprocess.run(
                ["git", "push", "origin", "HEAD:perf-history"], cwd=hist
            ).returncode
            == 0
        ):
            break
        print(f"push attempt {attempt} rejected; retrying from a fresh clone")
    else:
        sys.exit("could not push to perf-history")

    run("gh", "workflow", "run", "generateDocs.yml", "--ref", "main", cwd=REPO)
    print("docs rebuild requested")
    return 0


if __name__ == "__main__":
    sys.exit(main())
