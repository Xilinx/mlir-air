# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Q4NX Llama-3.2-1B example.

The NPU runs the FastFlowLM 4-bit `model.q4nx` weights (dequant'd to bf16); the
reference is the HF **bf16** checkpoint. The shared verify gate compares
NPU-q4nx vs HF-bf16 (top-k token-set inclusion) — the same NPU-vs-bf16 proxy the
3B Q4NX adapter uses.

Unlike the 3B Q4NX adapter (which reuses the bf16 llama32_3b `NpuRunner`
verbatim), the 1B decode is the fused superkernel: prefill runs through
`LlamaQ4nxPrefill` and each decode step through the one-xclbin `FusedDecoder`.
So this adapter drives those two components directly to satisfy the shared
`prefill()` / `decode_step()` Runner contract.

Pointed at via `--runner=llama32_1b_q4nx.verify_adapter`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_VERIFY = _LLMS_DIR / "verify"
_LLAMA1B = _LLMS_DIR / "llama32_1b"
for _p in (str(_LLMS_DIR), str(_VERIFY), str(_LLAMA1B), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

# Q4NX Llama-3.2-1B shares the bf16 1B architecture (16 layers, emb=2048,
# 32Q/8KV, head_dim=64); reuse its config so the shared HfRunner sees the dims
# it needs (n_layers / emb_dim / n_kv_heads / head_dim).
from llama32_1b_weights import LlamaConfig  # noqa: E402

MODEL_CHOICES = {
    "base": "meta-llama/Llama-3.2-1B",
    "instruct": "meta-llama/Llama-3.2-1B-Instruct",
}
DEFAULT_MODEL = "instruct"

# NPU weight source (model.q4nx bundle). `model_name` selects the tokenizer + HF
# bf16 reference; the NPU weights always come from here.
Q4NX_MODEL_SOURCE = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Llama-3.2-1B-NPU2")

_N_LAYERS = 16


def build_config():
    return LlamaConfig()


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """Reference is the HF bf16 checkpoint (NPU q4nx vs HF bf16)."""
    return npu_model_name


def build_runner(
    model_name: str,
    config,
    max_seq: int,
    tokenizer,
    *,
    npu_attn: bool = True,
    lite_mode: bool = False,
):
    """Compile/load the Q4NX prefill + fused decode kernels, return an NpuRunner."""
    return NpuRunner(max_seq=max_seq, tokenizer=tokenizer, lite_mode=lite_mode)


def _stair_on():
    """Multi-window decode opt-in; same env switch as the CLI."""
    return os.environ.get("DECODE_STAIRCASE") == "1"


class NpuRunner:
    """Adapter over the Q4NX fused prefill + one-xclbin decode.

    prefill() runs `LlamaQ4nxPrefill` (full logits + per-layer KV) and seeds the
    decoder's region-major KV cache; decode_step() dispatches one fused token
    through `FusedDecoder`. Mirrors the resident pipeline in
    `llama32_1b_q4nx_inference.Session`."""

    name = "npu_q4nx"

    def __init__(self, max_seq: int, tokenizer, lite_mode: bool = False):
        self.max_seq = max_seq
        self.lite_mode = lite_mode
        self._tokenizer = tokenizer

        from llama32_1b_q4nx_prefill import LlamaQ4nxPrefill
        from llama32_1b_q4nx_inference import FusedDecoder

        self.prefiller = LlamaQ4nxPrefill(seq_len=max_seq, n_layers=_N_LAYERS)
        self.prefiller.load_weights(model=Q4NX_MODEL_SOURCE)
        self.dec = FusedDecoder(staircase=_stair_on())
        self.attn_maxl = self.dec.ATTN_MAXL
        self._P = 0

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        ids = [int(t) for t in prompt_tokens]
        self.prefiller.clear_context()
        logits = np.asarray(self.prefiller.prefill(ids), np.float32)
        first = int(logits.argmax())
        K = np.stack(
            [
                np.asarray(self.prefiller.kv_view(l)[0], np.float32)
                for l in range(_N_LAYERS)
            ]
        )
        V = np.stack(
            [
                np.asarray(self.prefiller.kv_view(l)[1], np.float32)
                for l in range(_N_LAYERS)
            ]
        )
        self._P = K.shape[1]
        # Reset + seed the decoder's region-major KV cache from this prefill.
        self.dec.KV[:] = 0
        self.dec.seed_kv(K, V, self._P)

        # Q4NX prefill doesn't expose per-layer ffn_out, so the diagnosis lens has
        # no data. Return one empty dict per layer (len == n_layers) rather than an
        # empty list: the diagnosis path indexes layer_intermediates[li] per layer,
        # and `.get("ffn_out")` on an empty dict yields None, which the shared
        # runner skips gracefully (an empty list would IndexError).
        empty = np.empty((0,), dtype=np.float32)
        return PrefillRecord(
            layer_intermediates=[{} for _ in range(_N_LAYERS)],
            final_hidden_normed=empty,
            logits_at_pred=logits,
            top1_token=first,
        )

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        logits = np.asarray(
            self.dec.dispatch(int(input_token), int(current_pos)), np.float32
        )
        return DecodeStepRecord(
            lm_head_logits=logits,
            top1_token=int(logits.argmax()),
        )
