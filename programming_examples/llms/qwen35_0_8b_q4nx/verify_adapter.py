# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for Qwen3.5-0.8B Q4NX (fused prefill + fused_decode_gdn).

Drives the same path `make run` uses (q35_npu.Qwen35Npu): the prompt on the
shared fused_prefill device, then one fused_decode_gdn dispatch per token (24
layers and the full-vocabulary lm_head), the prompt's KV rows, conv history
and DeltaNet states handed to the decode at its first step.

REFERENCE. FastFlowLM's bundle (FastFlowLM/Qwen3.5-0.8B-NPU2, branch
flm_q4k_high_precision) was quantized from Qwen/Qwen3.5-0.8B (see README.md,
"Correctness"). The gate compares the NPU against that checkpoint's BF16
forward at the pinned revision: resolve_model() returns its local snapshot,
so the tokenizer and the reference load exactly that revision.

The build directory and decode capacity come from the Makefile (Q35_BUILD,
Q35_CAPACITY). Pointed at via `--runner=qwen35_0_8b_q4nx.verify_adapter`.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_VERIFY = _THIS_DIR.parent / "verify"
if str(_VERIFY) not in sys.path:
    sys.path.insert(0, str(_VERIFY))

from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

# The model is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.qwen35_0_8b_q4nx import q35_npu  # noqa: E402

_N_LAYERS = 24


@dataclass
class Qwen35Config:
    n_layers: int = _N_LAYERS
    emb_dim: int = 1024
    n_heads: int = 8
    head_dim: int = 256
    n_kv_heads: int = 2
    hidden_dim: int = 3584
    vocab_size: int = 248320


# Qwen3.5-0.8B is the instruction-tuned checkpoint the bundle was built from;
# both keys map to it.
MODEL_CHOICES = {
    "base": q35_npu.HF_REPO,
    "instruct": q35_npu.HF_REPO,
}
DEFAULT_MODEL = "instruct"


def build_config():
    return Qwen35Config()


def resolve_model(model_choice_or_id: str) -> str:
    """The pinned revision's local snapshot (downloaded on first use)."""
    repo = MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)
    if repo != q35_npu.HF_REPO:
        return repo
    from huggingface_hub import snapshot_download

    # the files the runner's offline cache holds (utils/hf_disk_budget.py)
    return snapshot_download(
        repo,
        revision=q35_npu.HF_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )


def build_runner(
    model_name: str,
    config,
    max_seq: int,
    tokenizer,
    *,
    npu_attn: bool = True,
    lite_mode: bool = False,
):
    return NpuRunner(lite_mode=lite_mode)


class NpuRunner:
    """prefill() runs the fused prefill and returns the last row's logits;
    decode_step() dispatches one fused_decode_gdn token."""

    name = "npu_q4nx"

    def __init__(self, lite_mode: bool = False):
        self.lite_mode = lite_mode
        build = os.environ.get("Q35_BUILD") or str(_THIS_DIR / "build")
        capacity = int(os.environ.get("Q35_CAPACITY", "1024"))
        self.m = q35_npu.Qwen35Npu(build, capacity)

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        ids = [int(t) for t in prompt_tokens]
        logits = np.asarray(self.m.prefill(ids), np.float32)
        # The prefill keeps no per-layer probes and the decode runs all 24
        # layers in one dispatch: one empty dict per layer for the diagnosis
        # lens, which skips layers without data.
        return PrefillRecord(
            layer_intermediates=[{} for _ in range(_N_LAYERS)],
            final_hidden_normed=np.empty((0,), dtype=np.float32),
            logits_at_pred=logits,
            top1_token=int(logits.argmax()),
        )

    def prefill_logits(self, prompt_tokens: np.ndarray, logits_from: int):
        """Logits of the prompt's rows from logits_from on, from one prefill
        (accuracy.py --path prefill, lm_eval_npu.py loglikelihood)."""
        ids = [int(t) for t in prompt_tokens]
        return np.asarray(self.m.fp.prefill(ids, logits_from=logits_from), np.float32)

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        assert current_pos == self.m.pos, (current_pos, self.m.pos)
        logits = np.asarray(self.m.decode(int(input_token)), np.float32)
        return DecodeStepRecord(
            lm_head_logits=logits,
            top1_token=int(logits.argmax()),
        )
