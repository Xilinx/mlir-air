# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Sequential, gate-before-profile CI for one native affine-Q4 configuration.

Each invocation retains its own reference, captures, logs and status. Packed
weights are shared within a source-bound work directory; compilation still runs
through the normal dependency-aware build. The profile target is self-contained.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from .deployment import LLMS, MODELS, configuration, file_hash, save_json
from .deployment_cli import device_lock


def source_identity():
    root = LLMS.parents[1]
    names = subprocess.check_output(
        [
            "git",
            "ls-files",
            "-co",
            "--exclude-standard",
            "programming_examples",
            "python/air",
        ],
        cwd=root,
        text=True,
    ).splitlines()
    digest = hashlib.sha256()
    for name in sorted(set(names)):
        path = root / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + path.read_bytes())
    return dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        source_sha256=digest.hexdigest(),
    )


def perf_id(model, selected):
    return f"{model}_p{selected['tokens']}_c{selected['capacity']}"


def performance_record(model, selected, timing):
    """Validate the measured protocol and compute metrics from raw samples."""
    if (
        timing["model"] != model
        or timing["configuration"] != selected
        or timing["outputs"] != 128
    ):
        raise ValueError("timing model/configuration/output identity differs")
    warmups, requests = timing["warmups"], timing["requests"]
    for row in warmups + requests:
        if len(row["tokens"]) != 128:
            raise ValueError("expected exactly 128 outputs per request")
        for field in ("ttft_seconds", "decode_seconds", "request_seconds"):
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError("invalid request duration: " + field)
        if not math.isclose(
            row["ttft_seconds"] + row["decode_seconds"],
            row["request_seconds"],
            rel_tol=1e-8,
            abs_tol=1e-8,
        ):
            raise ValueError("inconsistent request timing boundaries")
    if (
        sum(r["request_seconds"] for r in warmups) < 5
        or len(requests) < 3
        or sum(r["request_seconds"] for r in requests) < 10
    ):
        raise ValueError("incomplete warmup/measurement protocol")
    ttft = statistics.median(r["ttft_seconds"] for r in requests)
    return dict(
        model=perf_id(model, selected),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        metrics=dict(
            ttft_ms=1000 * ttft,
            prefill_tokens_per_sec=selected["tokens"] / ttft,
            decode_tokens_per_sec=127
            * len(requests)
            / sum(r["decode_seconds"] for r in requests),
            context_len=selected["tokens"],
            mean_request_ms=1000
            * statistics.mean(r["request_seconds"] for r in requests),
        ),
        run_params=dict(
            workload="fixed full-input prefill; fresh complete requests",
            model_module=model,
            prefill_length=selected["tokens"],
            kv_capacity=selected["capacity"],
            input_ids_sha256=selected["ids_sha256"],
            n_tokens=128,
            ignore_eos=True,
            measured_requests=len(requests),
            prefill_gemm="BFP16",
            ttft_boundary=timing["ttft_boundary"],
        ),
        samples=dict(warmups=warmups, requests=requests),
    )


def run_command(argv, log):
    print("[affine-ci]", log.stem, flush=True)
    with log.open("w") as stream:
        stream.write(repr(argv) + "\n")
        stream.flush()
        result = subprocess.run(
            argv,
            cwd=LLMS,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1"),
        )
    if result.returncode:
        print(log.read_text()[-8000:], file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, argv)


def execute(args, run=run_command):
    selected = configuration(args.model, args.prefill)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    name = perf_id(args.model, selected)
    metric = work / (name + ".perf.json")
    # A failed rerun must not leave a success metric from an earlier invocation.
    metric.unlink(missing_ok=True)
    evidence = work / "evidence"
    evidence.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=name + "-" + args.stage + "-", dir=evidence))
    identity = source_identity()
    asset_key = dict(
        source=identity["source_sha256"],
        q4_manifest=(file_hash(args.q4_dir / "manifest.json") if args.q4_dir else None),
    )
    build_key = hashlib.sha256(
        json.dumps(asset_key, sort_keys=True).encode()
    ).hexdigest()
    build = work / "build" / args.model / build_key[:20]
    build.mkdir(parents=True, exist_ok=True)
    state = dict(
        model=args.model,
        configuration=selected,
        source=identity,
        stage=args.stage,
        status="RUNNING",
        evidence=str(out),
        commands=[],
    )
    save_json(out / "status.json", state)

    def call(command, extra=(), python=sys.executable):
        argv = [
            python,
            "-u",
            "-m",
            args.model,
            command,
            "--prefill",
            str(args.prefill),
            "--build-dir",
            str(build),
        ]
        if args.checkpoint:
            argv += ["--checkpoint", str(args.checkpoint.resolve())]
        if args.q4_dir:
            argv += ["--q4-dir", str(args.q4_dir.resolve())]
        argv += list(extra)
        entry = dict(command=command, argv=argv, status="RUNNING")
        state["commands"].append(entry)
        save_json(out / "status.json", state)
        start = time.monotonic()
        run(argv, out / (command + ".log"))
        entry.update(status="PASS", seconds=time.monotonic() - start)
        save_json(out / "status.json", state)

    try:
        # The source key changes on edits to tracked or new example sources.
        # Incomplete packing has no manifest and the CLI refuses to overwrite it.
        if not (build / "weights/manifest.json").is_file():
            call("weights")
        call("compile")
        call("reference", ["--out", str(out / "reference")], args.reference_python)
        extra = [
            "--out",
            str(out / "verify"),
            "--reference-dir",
            str(out / "reference"),
        ]
        if args.prefill == 512:
            extra.append("--check-reset")
        call("verify", extra)
        gate = json.loads((out / "verify/gate.json").read_text())
        if gate["status"] != "OK":
            raise ValueError("model gate failed; profiling is disabled")
        if args.prefill == 512:
            reset = json.loads((out / "verify/reset.json").read_text())
            if reset["bitwise_logits_equal"] is not True:
                raise ValueError("request-isolation check failed")
        if args.stage == "profile":
            call("profile", ["--out", str(out / "profile")])
            record = performance_record(
                args.model,
                selected,
                json.loads((out / "profile/timing.json").read_text()),
            )
            record.update(source=identity, gate=gate, evidence_directory=str(out))
            save_json(metric, record)
        state["status"] = "PASS"
        print(f"[affine-ci] PASS {name} {args.stage}", flush=True)
    except Exception as exc:
        state.update(status="FAIL", error=str(exc))
        raise
    finally:
        save_json(out / "status.json", state)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "profile"))
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--q4-dir", type=Path)
    parser.add_argument("--reference-python")
    args = parser.parse_args()
    key = (
        "AIR_OPENELM_REFERENCE_PYTHON"
        if args.model == "openelm_1b_q4nx"
        else "AIR_AFFINE_Q4_REFERENCE_PYTHON"
    )
    args.reference_python = args.reference_python or os.environ.get(key)
    if not args.reference_python:
        parser.error(f"set {key}; see shared/AFFINE_Q4_DEPLOYMENTS.md")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with device_lock(args.work_dir / ".ci.lock"):
        execute(args)


if __name__ == "__main__":
    main()
