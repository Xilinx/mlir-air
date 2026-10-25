# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare and cache Qwen3-8B NPU-prefill states for the shared fixed cohort."""

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
from paths import HERE, COMMON
from common.build_identity import digest
from model_paths import (
    TARGET_REVISION,
    TOKENIZER_REVISION,
    DRAFT_REVISION,
    load_tokenizer,
)

PROMPTS = COMMON / "benchmark/prompts.json"
REVISIONS = dict(
    target=TARGET_REVISION, tokenizer=TOKENIZER_REVISION, drafter=DRAFT_REVISION
)


def selected_cases(per_dataset):
    cohort = json.loads(PROMPTS.read_text())
    return cohort, [c for c in cohort["cases"] if c["selection_rank"] < per_dataset]


def verify_inputs(inputs):
    inputs = Path(inputs)
    actual = json.loads((inputs / "cases.json").read_text())
    if actual.get("status") != "prepared" or actual.get("model_revisions") != REVISIONS:
        raise ValueError("Incomplete or incompatible Qwen3-8B prompt states")
    if actual.get("prompts_sha256") != digest(PROMPTS):
        raise ValueError("Prompt cohort differs from the shared frozen manifest")
    _, expected = selected_cases(actual["samples_per_dataset"])
    if len(actual["cases"]) != len(expected):
        raise ValueError("Incomplete prompt cohort")
    for case, source in zip(actual["cases"], expected):
        if any(case.get(k) != v for k, v in source.items()):
            raise ValueError(
                "Prompt text, order or identity differs from the frozen cohort"
            )
        if digest(inputs / case["cache_file"]) != case["cache_sha256"]:
            raise ValueError(f"Prefill checksum mismatch: {case['id']}")
    return actual


def preparation_identity(artifacts):
    artifacts = Path(artifacts)
    manifest = json.loads((artifacts / "manifest.json").read_text())
    return dict(
        model_revisions=REVISIONS,
        prompts_sha256=digest(PROMPTS),
        source_sha256={
            str(p.relative_to(HERE.parents[1])): digest(p)
            for p in (
                HERE / "npu_prefill.py",
                HERE / "prefill_weights.py",
                HERE / "model_spec.py",
                COMMON / "prefill_weights_8b.py",
                COMMON / "prefill_cache.py",
                HERE.parents[1] / "qwen3_8b_q4nx/qwen3_8b_q4nx_prefill.py",
                HERE.parents[1] / "qwen3_8b_q4nx/qwen3_8b_q4nx_weights.py",
                HERE.parents[1] / "shared/infra/cache.py",
            )
        },
        prefill_config=digest(artifacts / "dflash_prefill_config.json"),
        # Actual binaries avoid invalidation by incidental build timestamps or
        # absolute paths in the shared cache manifest.
        programs={
            name: digest(Path(entry["output_binary"]))
            for name, entry in manifest["entries"].items()
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=HERE / "build/benchmark")
    p.add_argument("--prefill-artifacts", type=Path)
    p.add_argument("--per-dataset", type=int, choices=(4, 16), default=16)
    p.add_argument("--verify", type=Path, metavar="INPUTS")
    a = p.parse_args()
    if a.verify:
        print(f"Verified {len(verify_inputs(a.verify)['cases'])} saved prompt states")
        return
    if a.prefill_artifacts is None:
        p.error("--prefill-artifacts is required to prepare NPU states")
    from npu_prefill import NpuPrefill

    cohort, cases = selected_cases(a.per_dataset)
    destination = a.destination.resolve()
    prefill = NpuPrefill(seq_len=2048, cache_dir=a.prefill_artifacts)
    identity = preparation_identity(a.prefill_artifacts)
    generation = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:20]
    inputs = destination / generation / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    manifest_path = inputs / "cases.json"

    def publish():
        # Previous source/build generations remain available for recorded studies.
        # Point new Make invocations at the current one only after validation.
        public = destination / "inputs"
        if public.exists() and not public.is_symlink():
            raise ValueError(
                "Legacy prompt directory at destination; select a fresh BENCHMARK_BUNDLE"
            )
        temporary = destination / "inputs.partial"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(Path(generation) / "inputs")
        temporary.replace(public)

    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if prior and (
        prior.get("preparation") != identity
        or prior.get("prompts_sha256") != digest(PROMPTS)
    ):
        raise ValueError(
            "Saved prefill source/build differs; choose a fresh destination"
        )
    completed = {c["id"]: c for c in prior["cases"]} if prior else {}
    # A cached Full80 cohort also serves Full20 without rewriting its manifest.
    if (
        prior
        and prior.get("status") == "prepared"
        and prior["samples_per_dataset"] >= a.per_dataset
    ):
        verify_inputs(inputs)
        publish()
        print(f"Reusing {len(prior['cases'])} verified model-specific prompt states")
        return
    record = dict(
        status="preparing",
        model="Qwen3-8B",
        model_revisions=REVISIONS,
        prompts_sha256=digest(PROMPTS),
        samples_per_dataset=a.per_dataset,
        dataset_order=cohort["dataset_order"],
        preparation=identity,
        prefill="NPU Q4NX operators; C2048 padded; only real prompt rows saved",
        cases=[],
    )

    def save():
        tmp = manifest_path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(manifest_path)

    tokenizer = load_tokenizer()
    loaded = False
    for source in cases:
        text = tokenizer.apply_chat_template(
            source["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        filename = source["id"].replace(":", "_") + ".npz"
        path = inputs / filename
        old = completed.get(source["id"])
        if old:
            if old["input_ids"] != ids or digest(path) != old["cache_sha256"]:
                raise ValueError(f"Changed saved input: {source['id']}")
            row = old
        else:
            if path.exists():
                raise FileExistsError(f"Unrecorded prompt state: {path}")
            if not loaded:
                prefill.load_weights()
                loaded = True
            result, seconds = prefill.prepare(ids)
            with path.with_suffix(".npz.partial").open("wb") as stream:
                np.savez(
                    stream,
                    input_ids=np.asarray(ids, np.int64),
                    first=result["first"],
                    **{
                        k: np.asarray(result[k], bfloat16).view(np.uint16)
                        for k in ("k", "v", "taps")
                    },
                )
            path.with_suffix(".npz.partial").replace(path)
            row = dict(
                source,
                input_ids=ids,
                cache_file=filename,
                cache_sha256=digest(path),
                prefill_seconds=seconds,
            )
        record["cases"].append(row)
        save()
        print(
            f"Prepared {len(record['cases'])}/{len(cases)}: {source['id']} ({len(ids)} tokens)",
            flush=True,
        )
    record["status"] = "prepared"
    save()
    verify_inputs(inputs)
    publish()


if __name__ == "__main__":
    main()
