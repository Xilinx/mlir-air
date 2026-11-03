# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""verify/ adapter for the affine-Q4 examples on the fused prefill.

A model's verify_adapter.py binds its name with adapter(). The runner is the
production session of the native configuration whose KV capacity covers the
runner's max_seq: the fused prefill for the prompt, its KV rows handed to the
fused decode, then one decode submission per token. The reference is the
pinned checkpoint; `make accuracy ACC_REF=dequant` replaces every matrix the
NPU runs in Q4 by its logical affine-Q4 values, dequantized.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import os
import sys

import numpy as np

from .deployment import LLMS, build_root, configuration, read_task, runtime, spec


def _records():
    verify = str(LLMS / "verify")
    if verify not in sys.path:
        sys.path.insert(0, verify)
    from runners._records import DecodeStepRecord, PrefillRecord

    return PrefillRecord, DecodeStepRecord


def checkpoint_dir(model):
    """The pinned checkpoint (deployment_cli.checkpoint), as a local path."""
    from .deployment_cli import checkpoint

    return str(checkpoint(model, os.environ.get("AIR_MODEL_CHECKPOINT")))


@dataclass
class _Config:
    n_layers: int
    emb_dim: int
    n_heads: int
    head_dim: int
    n_kv_heads: int
    hidden_dim: int
    vocab_size: int


def build_config(model):
    c = importlib.import_module(f"air_examples.llms.{model}.common").Config()
    return _Config(
        c.n_layers,
        c.emb_dim,
        c.n_heads,
        c.head_dim,
        c.n_kv_heads,
        c.hidden_dim,
        c.vocab_size,
    )


def build_dequant_hf_model(model, npu_model_name, hf_ref_model, config):
    """The checkpoint with each Q4 matrix replaced by its logical affine-Q4
    values. The embedding stays the checkpoint's bf16. When the model ties its
    head to the embedding, the head gets its own Q4 copy, so the reference is
    untied to hold both."""
    from .affine_q4 import CanonicalWeights

    verify = str(LLMS / "verify")
    if verify not in sys.path:
        sys.path.insert(0, verify)
    from dequant_reference import dequantized_reference

    q4 = CanonicalWeights(read_task(model)["assets"]["q4_root"])
    weights = [
        (name, lambda shape, n=name: q4.dequant(n).reshape(shape))
        for name, e in q4.entries.items()
        if e["kind"] == "affine_q4_g32"
    ]
    return dequantized_reference(hf_ref_model, weights)


class Runner:
    """The production session of one native configuration."""

    name = "npu_affine_q4"

    def __init__(self, model, max_seq):
        s = spec(model)
        configs = sorted(s["configurations"], key=lambda pc: pc[1])
        p = next((p for p, c in configs if c >= max_seq), configs[-1][0])
        selected = configuration(model, p)
        self.n_layers = build_config(model).n_layers
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        work = build_root(model) / "runs" / f"{stamp}-verify-adapter"
        factory = importlib.import_module(
            f"air_examples.llms.{model}.session"
        ).create_session
        print(
            f"[verify] {model}: fused prefill + decode of configuration "
            f"{selected['id']} (KV capacity {selected['capacity']})",
            flush=True,
        )
        self.session = factory(
            task=read_task(model, selected), runtime=runtime(model, selected, work)
        )

    def prefill(self, prompt_tokens):
        PrefillRecord, _ = _records()
        logits = self.session.prefill(np.asarray(prompt_tokens).reshape(-1))
        self.session.finish_prefill()
        # The decode runs every layer in one submission: no per-layer data.
        return PrefillRecord(
            layer_intermediates=[{} for _ in range(self.n_layers)],
            final_hidden_normed=np.empty(0, np.float32),
            logits_at_pred=logits,
            top1_token=int(np.argmax(logits)),
        )

    def prefill_logits(self, prompt_tokens, logits_from):
        """Logits of the prompt's rows from logits_from on, from one prefill."""
        return self.session.prefill_logits(
            np.asarray(prompt_tokens).reshape(-1), logits_from
        )

    def decode_step(self, input_token, current_pos):
        _, DecodeStepRecord = _records()
        logits = np.asarray(
            self.session.decode_step(int(input_token), int(current_pos)), np.float32
        )
        return DecodeStepRecord(lm_head_logits=logits, top1_token=int(logits.argmax()))


def adapter(model):
    """The module-level names verify_runner.py, accuracy.py and lm_eval_npu.py
    read from a verify adapter, for `model`."""
    os.environ.setdefault("AIR_TARGET_DEVICE", "npu2")
    choices = {"instruct": spec(model)["checkpoint"]}

    def resolve_model(choice):
        # The pinned revision, not the hub's current one.
        name = choices.get(choice, choice)
        return checkpoint_dir(model) if name == spec(model)["checkpoint"] else name

    def build_runner(model_name, config, max_seq, tokenizer, **_):
        return Runner(model, max_seq)

    def dequant(npu_model_name, hf_ref_model, config):
        return build_dequant_hf_model(model, npu_model_name, hf_ref_model, config)

    return dict(
        MODEL_CHOICES=choices,
        DEFAULT_MODEL="instruct",
        resolve_model=resolve_model,
        hf_reference=lambda name: name,
        build_config=lambda: build_config(model),
        build_runner=build_runner,
        build_dequant_hf_model=dequant,
    )
