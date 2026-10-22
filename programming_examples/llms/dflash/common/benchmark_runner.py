# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the 16 x 5 saved-prefill decode benchmark with the DFlash loop."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from common.speculative import autoregressive


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(rows):
    emitted = sum(r["emitted"] for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    official = sum(r["official_seconds"] for r in rows)
    rounds = sum(r["rounds"] for r in rows)
    return dict(
        requests=len(rows),
        emitted=emitted,
        seconds=seconds,
        tokens_per_second=emitted / seconds if seconds else None,
        official_tokens_per_second=emitted / official if official else None,
        equal_request_tokens_per_second=sum(r["tokens_per_second"] or 0 for r in rows)
        / len(rows),
        emitted_per_round=emitted / rounds if rounds else None,
        acceptance_rate=(
            sum(r["accepted_drafts"] for r in rows) / (7 * rounds) if rounds else None
        ),
    )


def main(
    *,
    verify_inputs,
    load_tokenizer,
    load_prefill,
    runtime_identity,
    load_pair,
    load_baseline,
    eos_ids=(151643, 151645),
):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "target", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("draft", "prepass"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--engine", choices=("dflash", "b1"), default="dflash")
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--per-dataset", type=int, choices=range(1, 17), default=16)
    a = p.parse_args()
    if a.engine == "dflash" and (a.draft is None or a.prepass is None):
        p.error("DFlash requires --draft and --prepass")
    if a.tokens < 2:
        p.error("--tokens must be at least 2")
    cohort = verify_inputs(a.inputs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse accidental mixing of runs. Each invocation gets its own file.
    stream = a.output.open("x")
    metadata = dict(
        runtime=runtime_identity(a.engine),
        status="loading",
        source_sha256=sha(__file__),
        inputs_sha256=sha(a.inputs / "cases.json"),
        manifests={
            name: sha(getattr(a, name) / "manifest.json")
            for name in (
                ("target", "draft", "prepass") if a.engine == "dflash" else ("target",)
            )
        },
        max_new_tokens=a.tokens,
        per_dataset=a.per_dataset,
        batch=8 if a.engine == "dflash" else 1,
        engine=a.engine,
        thinking=False,
        temperature=0,
        timing=(
            "Decode excludes saved prefill, reset, warmup and its first token; "
            + (
                "includes initial and later NPU context preparation, drafting, verification and host work."
                if a.engine == "dflash"
                else "includes every B1 forward and host token selection."
            )
        ),
        official_timing=(
            "Same emitted tokens; starts immediately before the first verification, after the first preparation and draft."
            if a.engine == "dflash"
            else "Identical to the primary decode timer."
        ),
    )
    summary_path = a.output.with_suffix(".summary.json")
    rows = []

    def save():
        groups = defaultdict(list)
        for row in rows:
            groups[row["dataset"]].append(row)
        result = dict(
            metadata,
            completed=len(rows),
            overall=aggregate(rows) if rows else None,
            per_dataset={name: aggregate(group) for name, group in groups.items()},
        )
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(summary_path)

    save()
    try:
        tokenizer = load_tokenizer()
        if a.engine == "dflash":
            loop = load_pair(a.target, a.draft, prepass_artifacts=a.prepass)
        else:
            target = load_baseline(a.target)
            if hasattr(target, "weight_identity"):
                metadata["baseline_weights"] = target.weight_identity
        counts = defaultdict(int)
        cases = []
        for case in cohort["cases"]:
            if counts[case["dataset"]] < a.per_dataset:
                cases.append(case)
                counts[case["dataset"]] += 1
        if any(counts[name] != a.per_dataset for name in cohort["dataset_order"]):
            raise ValueError("Insufficient prepared inputs for the requested cohort")
        initial = load_prefill(a.inputs, cases[0])
        if a.engine == "dflash":
            loop.prepass.warmup()
            loop.reset(initial)
            loop.generate(17, eos_ids=eos_ids)
        else:
            autoregressive(target, initial, 17, eos_ids=eos_ids)
        metadata["status"] = "running"
        for case in cases:
            prefill = load_prefill(a.inputs, case)
            if a.engine == "dflash":
                loop.reset(prefill)
                g = loop.generate(a.tokens, eos_ids=eos_ids)
            else:
                g = autoregressive(target, prefill, a.tokens, eos_ids=eos_ids)
            emitted = len(g.output_ids) - 1
            nr = len(g.rounds)
            row = dict(
                id=case["id"],
                dataset=case["dataset"],
                input_tokens=len(prefill["input_ids"]),
                prefill_sha256=sha(a.inputs / case["cache_file"]),
                emitted=emitted,
                rounds=nr,
                seconds=g.wall_seconds,
                official_seconds=g.official_seconds,
                initial_draft_seconds=g.initial_draft_seconds,
                tokens_per_second=emitted / g.wall_seconds if g.wall_seconds else None,
                emitted_per_round=emitted / nr if nr else None,
                accepted_drafts=sum(r["accepted_drafts"] for r in g.rounds),
                stages=g.stages,
                trace=g.rounds,
                output_ids=g.output_ids,
                output_text=tokenizer.decode(g.output_ids),
                stopped_on_eos=g.stopped_on_eos,
            )
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            save()
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "id",
                            "emitted",
                            "tokens_per_second",
                            "emitted_per_round",
                        )
                    }
                ),
                flush=True,
            )
        metadata["status"] = "completed"
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        save()
        stream.close()
    print(json.dumps(aggregate(rows), indent=2))
