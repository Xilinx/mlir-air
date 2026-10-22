# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Repeated NPU prefill, then DFlash against its sequential oracle and committed KV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.model import DFlashModel
from common.runtime import load_pair, load_pipeline, prepare_prompt, tokenize_prompt
from common.speculative import autoregressive


def signature(generation) -> list[dict]:
    return [
        {key: value for key, value in row.items() if key != "wall_seconds"}
        for row in generation.rounds
    ]


def main(model: DFlashModel) -> None:
    here = model.directory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Introduce me what is GPU")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument(
        "--prefill",
        choices=("npu", "cpu") if model.cpu_prefill else ("npu",),
        default="npu",
    )
    parser.add_argument(
        "--prefill-context", type=int, choices=(2048, 4096), default=2048
    )
    parser.add_argument("--prefill-artifacts", type=Path)
    parser.add_argument("--target-artifacts", type=Path, default=here / "build/target")
    parser.add_argument("--draft-artifacts", type=Path, default=here / "build/draft")
    parser.add_argument("--target-model", default=model.target_model)
    parser.add_argument("--tokenizer-model", default=model.tokenizer_model)
    parser.add_argument(
        "--target-cache", type=Path, default=model.default_target_cache()
    )
    parser.add_argument("--draft-cache", type=Path, default=model.default_draft_cache())
    parser.add_argument(
        "--draft-snapshot", type=Path, default=model.default_draft_snapshot()
    )
    parser.add_argument("--output", type=Path, default=here / "build/verify.json")
    args = parser.parse_args()
    result = {"passed": False}

    tokenizer, input_ids = tokenize_prompt(model, args.prompt, args.tokenizer_model)
    if args.prefill == "npu":
        loop, engine = load_pipeline(
            model,
            args.target_artifacts,
            args.draft_artifacts,
            target_model=args.target_model,
            target_cache=args.target_cache,
            draft_cache=args.draft_cache,
            draft_snapshot=args.draft_snapshot,
            prefill_context=args.prefill_context,
            prefill_artifacts=args.prefill_artifacts,
        )
    else:
        loop = load_pair(
            model,
            args.target_artifacts,
            args.draft_artifacts,
            target_model=args.target_model,
            target_cache=args.target_cache,
            draft_cache=args.draft_cache,
            draft_snapshot=args.draft_snapshot,
        )
    if len(input_ids) + args.tokens - 1 + loop.batch > loop.capacity:
        raise ValueError("Prompt plus verification output exceed KV capacity")
    result["block_size"] = loop.batch
    if args.prefill == "npu":
        prefill, prefill_seconds = engine.prepare(input_ids)
        repeated, _ = engine.prepare(input_ids)
        assert repeated["first"] == prefill["first"]
        for name in ("k", "v", "taps"):
            assert np.isfinite(prefill[name].astype(np.float32)).all()
            assert np.array_equal(
                repeated[name], prefill[name]
            ), f"NPU prefill repeat mismatch: {name}"
        # The standard attention PV uses BFP16 exponent groups over eight
        # key positions. Appending into a partial group can change rounding
        # inside that group; completed groups must remain exact. Record that
        # numerical boundary instead of claiming arbitrary-prefix bit equality.
        extended, _ = engine.prepare(input_ids + [13, 42, 123])
        complete = len(input_ids) // 8 * 8
        boundary = {}
        for name in ("k", "v", "taps"):
            a = prefill[name].astype(np.float32)
            b = extended[name].astype(np.float32)
            b = b[: len(input_ids)] if name == "taps" else b[:, : len(input_ids)]
            if name == "taps":
                assert np.array_equal(a[:complete], b[:complete]), name
            else:
                assert np.array_equal(a[:, :complete], b[:, :complete]), name
            boundary[name] = {
                "max_abs": float(np.max(np.abs(a - b))),
                "unequal": int(np.count_nonzero(a != b)),
            }
        # A new full attention tile must be invisible to the complete prefix.
        aligned_ids = input_ids + [13] * (-len(input_ids) % 64)
        aligned, _ = engine.prepare(aligned_ids)
        later, _ = engine.prepare(aligned_ids + [42, 123])
        for name in ("k", "v"):
            assert np.array_equal(
                aligned[name], later[name][:, : len(aligned_ids)]
            ), name
        assert np.array_equal(aligned["taps"], later["taps"][: len(aligned_ids)])
        result.update(
            npu_prefill_repeats_exact=True,
            npu_prefill_complete_groups_prefix_exact=True,
            npu_prefill_aligned_tile_prefix_exact=True,
            partial_group_prefix_difference=boundary,
            tap_slots=list(loop.tap_slots),
            tap_shape=list(prefill["taps"].shape),
            prefill_padded_length=engine.seq,
        )
    else:
        prefill, prefill_seconds = prepare_prompt(
            loop.target, input_ids, loop.tap_slots
        )

    # Fixed-length state coverage even if the model emits EOS early.
    oracle = autoregressive(loop.target, prefill, args.tokens, eos_ids=())
    committed = len(prefill["input_ids"]) + max(0, len(oracle.output_ids) - 1)
    oracle_kv = loop.target.read_kv(committed)

    loop.reset(prefill)
    round_kv_checks = []

    def observe(row, active):
        end = row["position"] + row["emitted"]
        exact = np.array_equal(active.target.read_kv(end), oracle_kv[:, :, :end])
        round_kv_checks.append(dict(committed_end=end, exact=bool(exact)))
        assert exact, round_kv_checks[-1]

    first = loop.generate(args.tokens, eos_ids=(), observer=observe)
    first_kv = loop.target.read_kv(committed)
    loop.reset(prefill)
    second = loop.generate(args.tokens, eos_ids=())

    assert first.output_ids == oracle.output_ids
    assert second.output_ids == first.output_ids
    assert signature(second) == signature(first)
    assert np.array_equal(first_kv, oracle_kv)
    result.update(
        passed=True,
        input_tokens=len(input_ids),
        output_tokens=len(first.output_ids),
        output_text=tokenizer.decode(first.output_ids, skip_special_tokens=True),
        prefill_backend=args.prefill,
        prefill_seconds=prefill_seconds,
        cpu_prefill_seconds=prefill_seconds if args.prefill == "cpu" else None,
        dflash_repeats_exact=True,
        same_block_oracle_output_exact=True,
        committed_target_kv_exact=True,
        checked_rounds=len(first.rounds),
        round_kv_checks=round_kv_checks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
