# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Q4NX Qwen2.5-3B example (FLM-faithful prefill + decode).

Drives the SAME on-device path `make gen` uses: the AIR prefill for the prompt,
then the shared fused_decode superkernel (36 layers + the tied LM head in one
dispatch, attention + KV cache on the AIE array) for each new token.

REFERENCE. The NPU weights are FastFlowLM's pre-quantized Q4NX bundle for
`Qwen/Qwen2.5-3B-Instruct`, so the bf16 reference is that same checkpoint and the
gate measures the 4-bit loss and nothing else (top-k token-set inclusion, the
NPU-vs-bf16 proxy every Q4NX adapter uses).

PROMPT LENGTH. The fused decode grows a KV cache inside a build-time cap
(`LBUILD`): the device attends over the real context and appends at the real
position, so any prompt that fits the cap works and there is no minimum. The
gate therefore runs the shared `verify/prompts/*.txt` like the Llama adapters.

Pointed at via `--runner=qwen25_3b_q4.verify_adapter`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_VERIFY = _LLMS_DIR / "verify"
_QWEN3B = _LLMS_DIR / "qwen25_3b"  # shared prefill builders + config
for _p in (str(_LLMS_DIR), str(_VERIFY), str(_QWEN3B), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

# The Q4NX example is a 4-bit realization of the bf16 qwen25_3b architecture (36
# layers, emb=2048, 16Q/2KV, head_dim=128); reuse its config so the shared
# HfRunner sees the dims it needs.
from qwen25_3b_weights import LlamaConfig  # noqa: E402
import qwen25_3b_q4_weights as _w  # noqa: E402

# FastFlowLM quantized the *Instruct* checkpoint, so both keys map to it: a
# base-model reference would disagree with the weights it is checking.
MODEL_CHOICES = {
    "base": "Qwen/Qwen2.5-3B-Instruct",
    "instruct": "Qwen/Qwen2.5-3B-Instruct",
}
DEFAULT_MODEL = "instruct"

# NPU weight source: FastFlowLM's Q4NX bundle (a repo id, a local dir holding
# model.q4nx, or a direct path). Same default as the driver/Makefile. A
# full-precision HF checkpoint also works and is quantized on load.
QWEN_Q4_MODEL_SOURCE = os.environ.get(
    "Q4NX_MODEL_SOURCE",
    os.environ.get("QWEN_Q4_MODEL_SOURCE", "FastFlowLM/Qwen2.5-3B-Instruct-NPU2"),
)

_N_LAYERS = _w.NUM_LAYERS
# Padded prefill length. The qwen25_3b GEMM registry shapes exist at M=2048, so
# the whole prefill runs there (same constraint as the Llama Q4NX examples).
_SEQ_LEN = int(os.environ.get("QWEN_Q4_SEQ_LEN", "2048"))


def build_config():
    return LlamaConfig(n_layers=_N_LAYERS)


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """Reference is the HF bf16 checkpoint the bundle was quantized from."""
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
    """Compile/load the Q4NX prefill + the fused Qwen2.5-3B decode, return an NpuRunner."""
    return NpuRunner(tokenizer=tokenizer, lite_mode=lite_mode)


class NpuRunner:
    """Adapter over the Q4NX prefill + one-xclbin fused Qwen2.5-3B decode.

    prefill() runs `Qwen25Q4Prefill` (full logits + per-layer roped-K / raw-V) and
    seeds the decoder's KV cache; decode_step() dispatches one fused token (36
    layers + the tied LM head) through `FusedDecoder`. Mirrors the loop in
    `qwen25_3b_q4_inference.generate`."""

    name = "npu_q4nx"

    def __init__(self, tokenizer, lite_mode: bool = False):
        self.lite_mode = lite_mode
        self._tokenizer = tokenizer

        from qwen25_3b_q4_prefill import Qwen25Q4Prefill
        from qwen25_3b_q4_inference import FusedDecoder

        self.prefiller = Qwen25Q4Prefill(seq_len=_SEQ_LEN, n_layers=_N_LAYERS)
        self.prefiller.load_weights(model=QWEN_Q4_MODEL_SOURCE)
        # `make compile-decode` builds the decode_L<N> template pair into THIS
        # example's directory; anchor it here rather than depending on where the
        # runner was launched from.
        os.environ.setdefault("Q4NX_QWEN25_3B_DECODE_DIR", str(_THIS_DIR))
        self.dec = FusedDecoder(model=QWEN_Q4_MODEL_SOURCE)
        self.attn_maxl = self.dec.ATTN_MAXL
        self._P = 0

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        ids = [int(t) for t in prompt_tokens]
        if len(ids) > self.attn_maxl:
            raise SystemExit(
                f"prompt is {len(ids)} tokens but the fused Qwen2.5-3B decode KV cache "
                f"caps at {self.attn_maxl}. Rebuild with a larger cap "
                f"(`make compile-decode LBUILD=...`)."
            )
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
        self.dec.seed_kv(K, V, self._P)

        # The prefill produces the first token and the decode continues from it, so
        # no priming dispatch: the runner's first decode_step IS that token at
        # position P (same contract as the 7B sibling).

        # The prefill does not expose per-layer ffn_out, and the decode runs all 36
        # layers inside one dispatch, so the diagnosis lens has no per-layer data.
        # Return one empty dict per layer (len == n_layers) rather than an empty
        # list: the diagnosis path indexes layer_intermediates[li] per layer, and
        # `.get("ffn_out")` on an empty dict yields None, which the shared runner
        # skips gracefully (an empty list would IndexError).
        empty = np.empty((0,), dtype=np.float32)
        return PrefillRecord(
            layer_intermediates=[{} for _ in range(_N_LAYERS)],
            final_hidden_normed=empty,
            logits_at_pred=logits,
            top1_token=first,
        )

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        # `current_pos` is the absolute position of `input_token` (the runner
        # passes len(prompt) for the first generated token), which is what
        # dispatch() wants.
        logits = np.asarray(
            self.dec.dispatch(int(input_token), int(current_pos)), np.float32
        )
        return DecodeStepRecord(
            lm_head_logits=logits,
            top1_token=int(logits.argmax()),
        )
