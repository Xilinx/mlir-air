# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Prepare and cache NPU-prefill states for the shared fixed prompt cohort."""

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
from paths import HERE, COMMON
from common.build_identity import digest
import model_spec
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
        raise ValueError("Incomplete or incompatible prompt states for this model")
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
    llms = HERE.parents[1]
    return dict(
        model_revisions=REVISIONS,
        prompts_sha256=digest(PROMPTS),
        source_sha256={
            str(p.relative_to(llms)): digest(p)
            for p in (
                HERE / "npu_prefill.py",
                HERE / "target_runtime.py",
                HERE / "verify_b8.py",
                HERE / "layers_b8.py",
                HERE / "lm_head_b8.py",
                HERE / "model_b8.py",
                HERE / "model_spec.py",
                llms / "qwen3_coder_30b_a3b_q4k/qwen3_coder_30b_a3b_q4k_weights.py",
            )
        },
        programs={
            p.parent.name: digest(p) for p in sorted(artifacts.glob("*/air.elf"))
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=HERE / "build/benchmark")
    p.add_argument("--target", type=Path, help="the target's build")
    p.add_argument("--per-dataset", type=int, choices=(4, 16), default=16)
    p.add_argument("--verify", type=Path, metavar="INPUTS")
    a = p.parse_args()
    if a.verify:
        print(f"Verified {len(verify_inputs(a.verify)['cases'])} saved prompt states")
        return
    if a.target is None:
        p.error("--target is required to prepare NPU states")
    from npu_prefill import NpuPrefill

    cohort, cases = selected_cases(a.per_dataset)
    destination = a.destination.resolve()
    prefill = NpuPrefill(seq_len=2048)
    identity = preparation_identity(a.target)
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
        model="Qwen3-Coder-30B-A3B-Instruct",
        model_revisions=REVISIONS,
        prompts_sha256=digest(PROMPTS),
        samples_per_dataset=a.per_dataset,
        dataset_order=cohort["dataset_order"],
        preparation=identity,
        prefill="NPU verify pass, 8 tokens per dispatch; only real prompt rows saved",
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
            **model_spec.CHAT_OPTIONS,
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
                from target_runtime import CoderTarget

                CoderTarget(a.target)
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
