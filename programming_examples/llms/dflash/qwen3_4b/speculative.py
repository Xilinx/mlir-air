# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Greedy DFlash host loop over one target and one NPU drafter.

The target's committed prefix excludes the current anchor token. A round appends
the anchor and candidate prefix to target KV and emits matching draft tokens plus
one verifier token. The next context injection replaces those draft-cache rows
with projections of the corresponding committed target features.
"""

from dataclasses import dataclass, field
import time

import numpy as np
from ml_dtypes import bfloat16

EOS_IDS = (151643, 151645)


@dataclass
class Decision:
    accepted_drafts: int
    tokens: list[int]
    stopped_on_eos: bool


def accept_greedy(block, target_tokens, remaining, eos_ids=EOS_IDS):
    """Pure acceptance/bonus/clipping rule; block[0] is already emitted."""
    if len(block) != len(target_tokens) or not block or remaining < 0:
        raise ValueError("Invalid verification block or token budget")
    accepted = 0
    for proposed, verified in zip(block[1:], target_tokens):
        if int(proposed) != int(verified):
            break
        accepted += 1
    emitted = [int(x) for x in block[1 : accepted + 1]] + [int(target_tokens[accepted])]
    emitted = emitted[:remaining]
    stop = next((i for i, token in enumerate(emitted) if token in eos_ids), None)
    if stop is not None:
        emitted = emitted[: stop + 1]
    return Decision(accepted, emitted, stop is not None)


@dataclass
class Generation:
    output_ids: list[int]
    wall_seconds: float
    rounds: list[dict] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    stopped_on_eos: bool = False
    official_seconds: float = 0.0
    initial_draft_seconds: float = 0.0


class DFlash:
    def __init__(self, target, drafter, prepass, tap_slots, mask_token_id):
        if (
            target.config.role != "target"
            or not target.config.causal
            or not target.config.taps
        ):
            raise ValueError(
                "Verification requires a causal target with hidden capture"
            )
        if drafter.config.role != "drafter" or drafter.config.causal:
            raise ValueError("Drafting requires the bidirectional drafter artifact")
        if target.batch != drafter.batch or target.capacity != drafter.capacity:
            raise ValueError(
                "Target and drafter must have matching block size and capacity"
            )
        self.target, self.drafter, self.prepass = target, drafter, prepass
        self.tap_slots, self.mask = list(tap_slots), int(mask_token_id)
        self.batch, self.capacity = target.batch, target.capacity

    def reset(self, prefill):
        self.prefill = prefill
        self.position = len(prefill["input_ids"])
        self.target.seed_kv(prefill["k"], prefill["v"], self.position)
        # Initial KV zeroing is setup. Learned context projection/upload remains
        # inside generation timing, including the first round.
        self.drafter.restore_kv(np.zeros(self.drafter.kv_shape, np.uint16))

    def generate(self, max_new_tokens, *, eos_ids=EOS_IDS, observer=None):
        if max_new_tokens < 0:
            raise ValueError("Negative token limit")
        if self.position + max(0, max_new_tokens - 1) + self.batch > self.capacity:
            raise ValueError(
                "Capacity cannot cover output limit and final verification block"
            )
        output = [int(self.prefill["first"])] if max_new_tokens else []
        if not output or output[-1] in eos_ids or max_new_tokens == 1:
            return Generation(
                output, 0.0, stopped_on_eos=bool(output and output[-1] in eos_ids)
            )
        pending = np.asarray(self.prefill["taps"], np.float32)
        pending_position = 0
        position = self.position
        stages = {
            key: 0.0
            for key in ("prepass", "kv_upload", "draft", "target", "tap_read", "accept")
        }
        rounds = []
        stopped = False
        start = time.perf_counter()
        official_start = None
        while len(output) < max_new_tokens and not stopped:
            round_start = time.perf_counter()
            t = time.perf_counter()
            _, k, v = self.prepass.run(pending, np.arange(pending_position, position))
            stages["prepass"] += time.perf_counter() - t
            t = time.perf_counter()
            self.drafter.append_context_kv(k, v, pending_position)
            stages["kv_upload"] += time.perf_counter() - t
            t = time.perf_counter()
            draft = self.drafter.verify(
                [output[-1]] + [self.mask] * (self.batch - 1), position
            )
            block = [output[-1]] + draft.logits[1:].argmax(axis=1).tolist()
            stages["draft"] += time.perf_counter() - t
            t = time.perf_counter()
            if official_start is None:
                # Decode timing starts after the first complete draft,
                # immediately before verify, as in the z-lab DFlash benchmark.
                official_start = t
            verified = self.target.verify(block, position)
            stages["target"] += time.perf_counter() - t
            t = time.perf_counter()
            prediction = verified.logits.argmax(axis=1).tolist()
            decision = accept_greedy(
                block, prediction, max_new_tokens - len(output), eos_ids
            )
            emitted = len(decision.tokens)
            old_position = position
            position += emitted
            output.extend(decision.tokens)
            stopped = decision.stopped_on_eos
            stages["accept"] += time.perf_counter() - t
            # Fetch only the committed boundary rows, so rejected rows never
            # reach the next round's context features.
            t = time.perf_counter()
            if len(output) < max_new_tokens and not stopped:
                bits = self.target.hidden(self.tap_slots)[:, :emitted]
                pending = (
                    bits.view(bfloat16)
                    .transpose(1, 0, 2)
                    .reshape(emitted, -1)
                    .astype(np.float32)
                )
                pending_position = old_position
            stages["tap_read"] += time.perf_counter() - t
            row = {
                "position": old_position,
                "accepted_drafts": decision.accepted_drafts,
                "raw_tau": decision.accepted_drafts + 1,
                "proposed": self.batch - 1,
                "emitted": emitted,
                "block_ids": block,
                "target_tokens": prediction,
                "emitted_ids": decision.tokens,
                "stopped_on_eos": stopped,
                "wall_seconds": time.perf_counter() - round_start,
            }
            rounds.append(row)
            if observer is not None:
                observer(row, self)
        end = time.perf_counter()
        return Generation(
            output,
            end - start,
            rounds,
            stages,
            stopped,
            official_seconds=end - official_start,
            initial_draft_seconds=official_start - start,
        )


def autoregressive(decoder, prefill, max_new_tokens, *, eos_ids=EOS_IDS, observer=None):
    """B1 GEMV baseline, or the same-GEMM row-zero correctness control."""
    p = len(prefill["input_ids"])
    if p + max(0, max_new_tokens - 1) + decoder.batch > decoder.capacity:
        raise ValueError("Capacity cannot cover the requested generation")
    decoder.seed_kv(prefill["k"], prefill["v"], p)
    output = [int(prefill["first"])] if max_new_tokens else []
    start = time.perf_counter()
    while output and len(output) < max_new_tokens and output[-1] not in eos_ids:
        position = p + len(output) - 1
        sample = decoder.verify([output[-1]] + [13] * (decoder.batch - 1), position)
        output.append(int(sample.logits[0].argmax()))
        if observer is not None:
            observer(position, sample, decoder)
    elapsed = time.perf_counter() - start
    return Generation(
        output,
        elapsed,
        stopped_on_eos=bool(output and output[-1] in eos_ids),
        official_seconds=elapsed,
    )
